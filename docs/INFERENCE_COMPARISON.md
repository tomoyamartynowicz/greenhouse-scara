# Vergelijking van inference-snelheid

Er is geen universeel aantal samplerstappen dat zowel diffusion als flow matching
op gelijke kwaliteit brengt. Rapporteer solver, stappen, NFE (aantal netwerk-
evaluaties), gemiddelde/mediaan/p95-latency en taakprestatie. NFE alleen is geen
latencymaat: de encoder, U-Net-grootte en overige berekeningen verschillen.

## Wat de literatuur vergelijkt

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf)
beschrijft echte robotexperimenten met 100 trainingstijdstippen en 10 DDIM-inferentiestappen.
[FlowPolicy](https://arxiv.org/html/2412.04987v2), tabellen 1 en 2, rapporteert
runtime én succespercentage: DP en DP3 met NFE=10 tegenover FlowPolicy met NFE=1,
over drie seeds. Die flowpolicy is speciaal met consistency-loss getraind.
Onze gewone conditional flow matching reproduceert die methode en één-stapskwaliteit niet.
[Flow-Guided Policies](https://openaccess.thecvf.com/content/ICCV2025W/ACVR/papers/Jung_Flow-Guided_Policies_Overcoming_Diffusion_Limitations_for_Robust_Robot_Imitation_Learning_ICCVW_2025_paper.pdf)
onderzoekt de afweging tussen integratiestappen, latency en policykwaliteit.
De vergeleken werken onderbouwen een kwaliteits/snelheidscurve, geen vaste algemene
FM-instelling die je zonder validatie kunt overnemen.

## Voorgesteld protocol voor deze modellen

Dit onderstaande raster is onze experimentele keuze, geen literatuurstandaard:

1. Houd hardware, precisie, batch 1, inputvormen, actiehorizon en meetgrenzen vast.
   Gebruik voor DP/FM dezelfde RGB-invoer, en voor DP3/FM3 dezelfde pointclouds.
   Bij ACT en de 3D-modellen verschillen architectuur en input; vermeld dat expliciet.
2. Meet DDIM en Euler bij NFE `[1,2,4,8,10,16,32]`. Meet Heun bij stappen
   `[1,2,4,5,8,16]`, dus NFE `[2,4,8,10,16,32]`. ACT heeft geen sampler-sweep.
   Deze rasterpunten passen binnen de huidige 100 DP-trainingstijdstippen.
3. Gebruik dezelfde vooraf geladen observaties en seed per instelling, warm-up,
   CUDA-synchronisatie en meerdere meetherhalingen. Zet andere GPU-belasting uit.
   Voor een uiteindelijke rapportage bijvoorbeeld 20 warm-ups, 100 metingen en
   drie aparte herhalingen; dit zijn nu de notebookdefaults.
4. Nu: rapporteer uitsluitend rekentijd versus NFE voor smokemodellen.
   De notebook voert het raster uit `NFE_BUDGETS` automatisch per model uit,
   inclusief beide flow-solvers. Elke modelcel start een eigen Python-proces.
   Het model en de input worden één keer geladen voor de hele sweep. JSON/CSV
   wordt na elke instelling bewaard; de laatste cellen tonen een tabel en curve.
5. Later: kies op een aparte validatieset per model/solver het laagste budget dat
   een vooraf vastgestelde taakprestatie haalt. Beoordeel daarna met vaste
   instellingen op testepisodes/robotrollouts en meerdere trainingsseeds.
   Actie-MSE alleen vervangt geen closed-loop taakprestatie.

DP=16 DDIM versus FM=8 Heun is één rijpaar uit de sweep met gelijke NFE=16,
maar garandeert geen gelijke kwaliteit of identieke latency. DP3=10 is ook
onderdeel van het raster. Publiceer geen algemene FM-speedup op
basis van arbitrair verschillende NFE zonder de kwaliteitsafweging erbij.

De benchmark meet vooraf geladen inputtensor -> volledige actiechunk. Resizing,
RGB-D-deprojectie, FPS, bestand-I/O en host/device-transfers zijn uitgesloten.
Voor een claim over volledige robotlatency moet die preprocessing apart of samen
met het model worden gemeten; met name volledige CPU-FPS kan veel tijd kosten.

## Geheugen en onderbreken

Herstart de kernel eenmalig na gebruik van de oude benchmark. Het notebook zelf
importeert geen PyTorch en houdt alleen numerieke resultaten vast. Iedere
modelcel start dezelfde Python-omgeving in een apart proces; na afloop verdwijnen
modelgewichten, CUDA-context en caches met dat proces. Checkpoints worden waar
mogelijk memory-mapped geladen; training/optimizer-tensors worden niet onnodig
ingelezen en vastgehouden tijdens inference. Dit vermindert de geheugenpiek,
maar een individueel model moet nog wel op de gekozen hardware passen.

Met Interrupt wordt het werkproces beëindigd. Bij een modelcrash kunnen de andere
cellen verder; reeds afgeronde instellingen staan in hun runmap. SIGKILL kan op
OOM wijzen, maar is zonder systeemlog geen bewijs van de oorzaak.

De sweep seed is gelijk per herhaling en wordt opnieuw ingesteld voor elke solver/
stapinstelling. De volgorde roteert per herhaling. De gerapporteerde p95 en mediaan
zijn berekend over alle ruwe metingen samen, niet als gemiddelde van percentielen.
CPU-RSS is de maximale proces-RAM sinds starten, inclusief laden; CUDA peak
allocated geldt per instelling inclusief model/input. Deze tellers meten geen
geheugengebruik van andere processen. Alle workers gebruiken float32, zonder TF32.
