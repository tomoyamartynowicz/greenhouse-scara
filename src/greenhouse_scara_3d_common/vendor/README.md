# Herkomst

Alleen de benodigde modelcode is overgenomen; geen simulators, task runners, datasets of checkpoints. De MIT-licenties staan bij de respectieve pakketten. Exacte commits en bestanden staan in [origins.json](origins.json).

- **DP3**: [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), commit `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`, uit `3D-Diffusion-Policy/diffusion_policy_3d`.
- **ManiFlow**: [allenai/maniflow](https://github.com/allenai/maniflow), commit `e4dc24d62a6c91825813308b9926e921b4bb9ef6`, uit `maniflow`.

Lokale wijzigingen aan deze bestanden:

- Imports verwijzen naar deze package-namespace; lege `__init__.py`-bestanden maken zelfstandig importeren mogelijk.
- De ongebruikte PyTorch3D-import in DP3 is verwijderd.
- ManiFlow importeert PyTorch3D alleen binnen zijn optionele FPS-functie. De SCARA-loader doet FPS al; `downsample_points=False` voorkomt dubbele sampling. Deze route heeft geen PyTorch3D nodig.
- ManiFlow `DP3Encoder.num_points` gebruikt de N-as van `[N, C]` voor correcte diagnostiek; met externe sampling verandert dit de berekening niet.

De officiële DP3 diffusion-loss en ManiFlow DiTX/flow-consistency-loss blijven behouden. ManiFlow krijgt tijdens training een EMA-model als teacher. De gedeelde SCARA-trainer, data-normalisatie, actieconstructie en puntenselectie zijn lokaal: dit is geen garantie op dezelfde benchmarkresultaten als upstream.

De nieuwe 3D-flowmap bevat daarnaast een minimale kopie van de bestaande `greenhouse_scara_flow_matching_policy`: dezelfde conditional U-Net, flow-loss en Euler/Heun-sampler, met aangepaste imports en de DP3-observatie-encoder. De 3D-ACT-map bevat de bestaande ACT `DETRVAE` en transformer, met een nieuwe PointNet-backbone; ongebruikte CNN- en bouwhelpers zijn weggelaten. De bijbehorende licenties staan in die projectmappen.
