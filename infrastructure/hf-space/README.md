---
title: KrishiGuard AI API
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Crop leaf disease detection API (EfficientNet-B0)
---

# KrishiGuard AI — backend API

FastAPI service for the KrishiGuard AI college project: crop-leaf disease classification (EfficientNet-B0, 38
PlantVillage classes), calibrated confidence, Grad-CAM explanations and English/Hindi advisory.

* Health: `/api/health` · Interactive docs: `/docs`
* The web frontend is deployed separately (Vercel) and proxies `/api/*` to this Space.

Decision support only — not a substitute for an agricultural expert.
