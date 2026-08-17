---
name: cvat-nuclio-model-deployment
description: Deploy, replace, and verify ONNX detector or OBB models in a Docker-based CVAT Nuclio installation. Use when a CVAT auto-annotation function must be added or updated, when a model handler/Dockerfile/function.yaml changes, or when deployment appears healthy but CVAT output is stale, has the wrong shape type, or violates an OBB angle convention.
---

# CVAT Nuclio Model Deployment

Use this skill for a complete deployment loop: inspect the existing function, build the image, deploy the same function, verify the live container, invoke the real HTTP endpoint, and check CVAT's function metadata. Treat source tests, container readiness, and CVAT behavior as separate gates.

## Inputs and safety

Collect these values from the repository and the running platform before changing anything:

- Nuclio source directory containing `function.yaml`, `Dockerfile`, handler code, and model file.
- Function name, project, namespace, platform (`local` or `kube`), and current node port from `nuctl get functions`.
- Model task (`obb` or axis-aligned detection), input size, class labels, and expected output schema.
- Docker network used by CVAT and the Redis host/port required by the deployment.

Preserve unrelated functions. Do not change PPE, SAM, or other CVAT functions unless explicitly requested. Never delete a model or container to resolve a deployment issue until its exact target and a recoverable alternative are identified.

Run the read-only preflight when the source follows the standard layout:

```bash
python3 scripts/preflight.py /path/to/nuclio/source
```

The preflight must report the model, handler, function name, labels, shape types, input size, and required tools before deployment proceeds.

## Deployment workflow

1. **Inspect and baseline.** Read `function.yaml`, `Dockerfile`, `main.py`, the model handler, and any local tests. Record the current function row, container image, port, and health state:

   ```bash
   nuctl get functions --platform local
   docker ps --format '{{.Names}}\t{{.Image}}\t{{.Ports}}'
   ```

   For a Kubernetes installation, use the configured kube context and `--platform kube` instead; do not mix local Docker and Kubernetes commands.

2. **Build the image.** Build from the function directory so all `COPY` paths are deterministic:

   ```bash
   docker build -t <image>:<tag> .
   ```

   Ensure the ONNX file copied into the image is the intended checkpoint. If the image is reused under a mutable tag such as `latest`, verify the container timestamp or code contents after deployment.

3. **Create a red-capable regression test before fixing output bugs.** Test at the handler seam with a fake session or a small fixture that drives the real post-processing. For OBB, include the equivalent representations `(w, h, theta)` and `(h, w, theta+90)` and assert both the serialized contract and unchanged corner geometry. Run the failing test before the fix, then the passing test after it:

   ```bash
   python3 -m unittest -v test_model_handler.py
   ```

4. **Deploy the same function.** Keep the existing function name and port mapping. A typical local Docker deployment is:

   ```bash
   nuctl deploy --project-name <project> --path . --file function.yaml \
     --platform local \
     --env CVAT_FUNCTIONS_REDIS_HOST=<redis-container> \
     --env CVAT_FUNCTIONS_REDIS_PORT=<redis-port> \
     --platform-config '{"attributes":{"network":"<cvat-network>"}}'
   ```

   Wait for the command to finish and require `Function deploy complete`; do not assume that a successful image build deployed the new code.

5. **Verify runtime state.** Require the target function to be `ready` with the expected replica count, and check the running container is healthy:

   ```bash
   nuctl get functions --platform local
   docker inspect --format '{{.State.Status}} {{.State.Health.Status}} {{.Config.Image}}' <container>
   docker exec <container> sh -c 'grep -n <distinctive-handler-symbol> /opt/nuclio/<handler-file>'
   ```

   If the distinctive symbol is missing, the container is stale even if Nuclio says `ready`.

6. **Invoke the live endpoint.** Use a real image encoded as base64 and the handler's actual request schema. Check HTTP status, output count, labels, shape types, confidence ranges, coordinates, and task-specific invariants. For example:

   ```bash
   base64 -w0 sample.jpg | jq -Rs '{image: ., threshold: 0.25}' \
     | curl --fail --silent --show-error \
       --header 'Content-Type: application/json' \
       --data-binary @- http://127.0.0.1:<node-port>
   ```

   Run this against representative normal, fallen/shifted, missing, edge, and empty scenes. Save the JSON or annotated images when visual review is needed.

7. **Verify CVAT registration.** Query the CVAT lambda API or the configured Nuclio gateway using the deployment's actual route and authentication. Confirm the function is visible, its IDs/names match `function.yaml`, and its shape types are correct. A direct Nuclio HTTP success is not proof that CVAT has refreshed its function metadata.

8. **Report completion with evidence.** Include changed files, test commands and results, image tag, function name, port, readiness state, live request result, CVAT metadata result, and any check that could not run. Remove temporary debug logs and generated caches before handoff.

## OBB output contract

Read [references/obb-and-cvat.md](references/obb-and-cvat.md) whenever the task involves rotated boxes, angles, polygons, or a model adapter. The short contract is:

- CVAT native rotated rectangles use `type: "rectangle"`, axis-aligned `points: [x1, y1, x2, y2]`, and `rotation`.
- Normalize the representation so the serialized rectangle's `width <= height`. If width and height are swapped, add `90` degrees. Then resolve the remaining `180`-degree ambiguity by keeping rotation in `[270, 360) or [0, 90)` so CVAT's local top edge stays above the center.
- Validate that the reconstructed four corners before and after normalization differ only by numerical tolerance.
- Do not use the CVAT width-edge angle as the cone posture angle. FALLEN/SHIFT business logic may intentionally use the polygon or rectangle long axis, which differs by about `90` degrees for an upright cone.
- Keep NMS on the model's original representation; normalize only at the serialized CVAT boundary unless the downstream algorithm explicitly requires the canonical form.

## Failure triage

Use [references/troubleshooting.md](references/troubleshooting.md) for symptoms and probes. In brief:

- `ready` but old behavior: compare image digest/timestamp and grep the live container for the new handler symbol.
- Function missing in CVAT: inspect the CVAT lambda route and Nuclio annotation `spec`; check that every shape is a CVAT-supported type.
- Polygon appears instead of rotated rectangle: fix the function annotation and returned `type`, then redeploy and refresh CVAT metadata.
- Angle is occasionally about `90` degrees wrong: test width/height canonicalization first. If the geometry is right but CVAT's top edge or rotation handle is upside down, canonicalize the equivalent angle by `180` degrees. Do not change business posture thresholds before proving the serialized corners are wrong.
- HTTP errors or model-load failures: inspect container logs, model path, ONNX runtime dependencies, request size, and input color/encoding.

Completion requires all four gates: regression tests pass, the target runtime is healthy with the new code, a real HTTP request satisfies the output contract, and CVAT reports the expected function metadata.
