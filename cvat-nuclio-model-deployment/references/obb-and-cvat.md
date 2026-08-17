# OBB and CVAT contract

## Model representation versus CVAT representation

Ultralytics/ONNX OBB output commonly contains `center_x, center_y, width, height, angle`. A rectangle can be represented equivalently as:

```text
(w, h, theta) == (h, w, theta + 90 degrees)
```

The model is free to emit either form. CVAT's native rotated rectangle should use one stable convention so human review and downstream rules do not jump by 90 degrees between frames.

## Canonical serialization

At the adapter boundary:

```python
if box_width > box_height:
    box_width, box_height = box_height, box_width
    rotation += 90.0
rotation %= 360.0
if 90.0 <= rotation < 270.0:
    rotation = (rotation + 180.0) % 360.0
```

Return:

```json
{
  "points": [x1, y1, x2, y2],
  "rotation": 5.2,
  "type": "rectangle"
}
```

`points` are the axis-aligned bounds around the center in CVAT image coordinates; `rotation` supplies the orientation. Do not replace this with a polygon unless the function's declared shape type and CVAT task explicitly require polygons.

The final half-plane normalization does not claim that the detector knows the cone's tip direction. A rectangle has no inherent arrow, so `theta` and `theta + 180 degrees` describe exactly the same geometry. This rule only makes CVAT place its local top edge and rotation handle consistently above the box center.

## Geometry regression

For every normalization test, reconstruct corners with the same convention used by the adapter (for example `cv2.boxPoints`). Sort the four points before comparison because corner ordering can differ. Require a small pixel tolerance. This catches the dangerous partial fix where width/height is exchanged without rotating the rectangle.

## Cone-specific angle meaning

CVAT `rotation` follows the canonical rectangle width edge. Cone posture logic should continue to use the cone's long axis or the project's established polygon-angle helper. Therefore an upright cone can have a CVAT width-edge angle near `0` degrees while its long-axis/posture angle is near `90` degrees. A fallen cone's alert angle must be compared with the business reference, not directly with CVAT's width-edge field.

## Verification checklist

- `type` is `rectangle` for every detector label declared as a rectangle.
- `0 <= rotation < 360` after serialization.
- `rotation` is in `[270, 360) or [0, 90)` after resolving the equivalent 180-degree representation.
- Width/height canonicalization is applied once, after NMS and coordinate restoration.
- Reconstructed corners match the pre-normalization geometry.
- Normal and fallen samples are visually reviewed; a numerically valid angle can still expose a model or coordinate-direction error.
