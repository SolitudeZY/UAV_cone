# Deployment troubleshooting

## Healthy function, stale output

1. Record the target function row from `nuctl get functions`.
2. Inspect the container image and status with `docker inspect`.
3. Grep a distinctive function symbol inside `/opt/nuclio`.
4. Check container startup logs for model loading and the handler path.
5. Rebuild with the intended source directory and redeploy the same function name.

Do not diagnose this from the source tree alone; Nuclio can be healthy while running an older processor image.

## CVAT does not show the function

Check the actual CVAT route configured by the deployment. In the standard CVAT application, lambda APIs are under `/api/lambda/functions`; Nuclio's own gateway uses `/api/functions` and requires the Nuclio project/namespace headers. A 404 at one route does not prove the other route is unavailable.

Inspect `function.yaml` annotations:

- `type: detector`
- JSON `spec` with stable integer IDs, labels, and CVAT-supported shape types
- the function name and namespace matching the deployed function

## Polygon instead of rotated rectangle

Make the annotation and response agree. For a native CVAT rotated rectangle, declare `type: "rectangle"` and return `type: "rectangle"` with `points` and `rotation`. Redeploy, then verify the metadata through CVAT and invoke the live endpoint again.

## Intermittent 90-degree angle error

First inspect raw `(width, height, angle)` rows and compare reconstructed corners with the framework's OBB corners. If corners match but width is sometimes the long edge, apply the canonical swap-plus-90 rule in `references/obb-and-cvat.md`. If all samples are mirrored, investigate coordinate-system direction. If corners disagree, investigate ONNX export/post-processing before changing CVAT semantics.

## CVAT rectangle appears upside down

Compare the reported rotation with the angle plus 180 degrees. If the reconstructed corners are identical, the geometry is already correct and only CVAT's local top edge/rotation handle is inverted. Normalize the equivalent angle into `[270, 360) or [0, 90)` at serialization. A standard OBB cannot identify the cone tip direction; use keypoints or an explicit appearance-direction classifier if semantic tip-versus-base orientation is required.

## HTTP or model-load failure

Check, in order:

- request JSON key and base64 image decoding;
- image color conversion and letterbox scale/padding;
- model path copied into the image;
- ONNX Runtime and OpenCV availability inside the image;
- request body limit and event timeout;
- container logs after the first request.

Keep the failing request fixture and its response status so the fix can be replayed.
