# Third-party dependencies and attribution

No third-party source checkout, wheel, vendor bundle, model weights or tokenizer
files are included. Static review of the selected 19 method modules and original
two replay modules found no explicit copied/adapted implementation notices or
third-party copyright headers. This is not proof of clean-room provenance.
The author's ownership confirmation covers authorized original work only.

Imported external libraries retain their own licenses and any third-party notices.
Verified upstream source families include PyTorch (BSD-style plus other notices),
TorchVision (BSD-3-Clause) and Transformers (Apache-2.0). Their exact historical
runtime versions are not fully pinned. Other imports are numpy, Pillow,
huggingface-hub, safetensors, scikit-learn, joblib, openai, neo4j and openpyxl;
installers must consult the particular distribution's license/NOTICE. No blanket
license clearance of all future versions or transitive dependencies is asserted.

Official pointers:
https://github.com/pytorch/pytorch/blob/main/LICENSE
https://github.com/pytorch/vision/blob/main/LICENSE
https://github.com/huggingface/transformers/blob/main/LICENSE
https://github.com/openai/CLIP/blob/main/LICENSE
https://docs.pytorch.org/vision/main/models.html

CLIP code MIT is not automatically treated as a verified weight redistribution
license. NLI model cards separately declare Apache-2.0 and base-model MIT; no
weights are distributed. DATA_SOURCES.md records original data attribution.
OpenEA, vis-network vendor bundles and unrelated project checkouts are not part
of this repository. Their licenses are not used to license this software.

Full MIT text is in LICENSE, with the SPDX standard template as its reference:
https://spdx.org/licenses/MIT.html
