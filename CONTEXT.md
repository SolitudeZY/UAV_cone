# Traffic Cone Candidate Review

This context defines the shared language for reviewing YOLO traffic-cone candidates before training the candidate truth and posture classifier.

## Language

**Detection candidate**:
A single object hypothesis produced by YOLO. Review decisions apply to this candidate, not to every object visible in the surrounding image.
_Avoid_: Whole image, scene label

**Review preview**:
An audit-only image that marks the detection candidate so a reviewer can identify the intended target. The marker is not part of the classifier's visual input.
_Avoid_: Classifier input, labeled training image

**Classifier crop**:
The unmarked, candidate-centered image used by the truth and posture classifier. The intended target must remain unambiguous without the review marker.
_Avoid_: Original frame, boxed image

**Non-fallen cone (`upright_cone`)**:
A real cone that is not visibly lying on the ground. This includes tilted, partially crushed, or supported cones that remain visibly standing.
_Avoid_: Perfectly vertical cone

**Fallen cone (`fallen_cone`)**:
A real cone whose visible body is clearly toppled or lying on the ground. Its projected image angle alone does not define this state.
_Avoid_: Horizontal box, angle-positive cone

**Non-cone (`not_cone`)**:
A candidate that can be confidently identified as background or another object rather than a traffic cone.
_Avoid_: Unclear cone, badly cropped cone

**Ignored candidate (`ignore`)**:
A review result for candidates whose identity, target, or posture cannot be labeled reliably. It is excluded from classifier training and evaluation and is not a fourth model class.
_Avoid_: Negative sample, uncertain class

**Unambiguous target**:
The single cone that remains identifiable as the candidate target in the classifier crop without relying on the review marker.
_Avoid_: Largest object in the crop, any visible cone
