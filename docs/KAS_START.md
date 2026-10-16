# Starten in de kas

Vooraf op de laptop: download de checkpoints en voer
`python scripts/test_2d_checkpoints.py` uit. De live-scripts gebruiken de
`current`-checkpointmappen onder `checkpoints/2d/`.

## ROBOT — dezelfde start als bij record_dataset.py

```bash
cd ~/Desktop/tomoya/scissor_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd scissor_runtime
python3 scissor_node.py
```

Laat deze terminal draaien. Gebruik de bestaande robot/TCS-bediening om de robot
in de juiste beginpose te zetten, zoals bij het opnemen. De policy-runner stuurt
J1–J4 via TCS en de schaar via de ROS2-node. Dezelfde `ROS_DOMAIN_ID` en
netwerkverbinding als bij het opnemen zijn nodig.

## EIGEN — één keer per terminal

```bash
conda activate act
source /opt/ros/humble/setup.bash
source ~/scara_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd ~/scara_ws/greenhouse-scara
```

## Eerst kijken, daarna uitvoeren

Bij `--execute` gebruikt de runner één TCS-verbinding en selecteert robot 1,
controleert `state`, schakelt zo nodig motorvermogen in (`hp 1 30`), en attacht
pas daarna. Vervolgens wordt bij iedere `--execute` standaard de SCARA gehomed.
Zet de pendant in Computer-modus. Bestaand vermogen/attachment wordt
gecontroleerd voordat het opnieuw wordt aangevraagd. Controllerfouten worden
niet automatisch gereset of opnieuw geprobeerd.

Homing kan de robot uit een handmatig ingestelde beginpose verplaatsen. Het
herstelt de referentie en brengt de robot niet automatisch terug naar die pose.
Je kunt eerst voorbereiden zonder daarna policy-acties te starten:

```bash
bash scripts/attempt.sh act_rgb voorbereiding --execute --home --prepare-only
```

Daarna kun je de robot in de demonstratiebeginpose zetten en voor die poging
`--skip-home` toevoegen om niet opnieuw te homen. Zonder die optie gebeurt homing
bij iedere uitgevoerde poging. Homing betreft de SCARA; de schaar wordt door zijn
eigen node geïnitialiseerd en door de runner voor iedere opname op nul gezet. Een `-3101`-fout
betekent servo-PID-saturatie, niet dat homing ontbreekt: controleer de foutlog en
mechanische situatie voordat je verdergaat. Bij `-1047`/`-1048` staat de pendant
nog in Jog-modus.

Eerste controle met ACT RGB: camerabeelden, gemeten standen en voorspellingen,
zonder bewegingscommando's. Deze controle wordt ook opgenomen:

```bash
bash scripts/attempt.sh act_rgb controle_01
```

Stop met `q` in het cameravenster of Ctrl+C in de terminal. Start daarna een
werkelijke poging (robot en schaar mogen bewegen):

```bash
bash scripts/attempt.sh act_rgb poging_01 --execute
```

Iedere uitgevoerde opname stuurt eerst de schaar naar **0 rad** en wacht totdat
de gemeten rol minimaal 0,3 seconde binnen ±0,02 rad staat. Daarna beginnen de
video-opname en policy-acties. Als dit niet binnen 10 seconden lukt of feedback
wegvalt, stopt de poging. Dit gebeurt automatisch voor alle modellen via
`attempt.sh ... --execute`; controles zonder `--execute` bewegen de schaar niet.
Bij stoppen wordt de beweging gestopt; de terugkeer naar nul gebeurt bij de
volgende opname, niet onverwacht tijdens het afsluiten.

Voor de volgende poging voer je opnieuw hetzelfde commando uit, bijvoorbeeld
met naam `poging_02`. Elke start krijgt automatisch een eigen tijdstempel, dus
een eerdere poging wordt nooit overschreven.

Kies één model per poging:

```bash
bash scripts/attempt.sh act_rgb poging_01 --execute
bash scripts/attempt.sh act_rgbd poging_01 --execute
bash scripts/attempt.sh dp_rgb poging_01 --execute
bash scripts/attempt.sh dp_rgbd poging_01 --execute
bash scripts/attempt.sh fm_rgb poging_01 --execute
bash scripts/attempt.sh fm_rgbd poging_01 --execute
```

ACT (RGB en RGBD) voert standaard de eerste **15 acties van de chunk van 30** uit,
neemt daarna nieuwe observaties en voorspelt opnieuw. DP/Flow gebruiken standaard
één actie per voorspelling; hun huidige checkpoints geven 8 uitvoerbare acties terug.
Met `--action-count N` kun je het aantal uitgevoerde acties aanpassen.

Voorspelde schaarrol wordt standaard geclipt naar `scissor_roll` in de configuratie
(momenteel −1,01 tot +1,01 rad). Een eindige voorspelling buiten dit bereik breekt
de poging niet af. De originele voorspelling blijft in de log. Ontbrekende,
ongeldige of buiten de grenzen gemeten schaarfeedback blijft een fout.

Bij `Action ... delta ... exceeds ...` kun je expliciet `--limit-action-steps`
toevoegen. De runner begrenst dan ieder doel tot de bestaande `max_delta` vanaf
de opnieuw gemeten stand, in plaats van wegens een te grote stap te stoppen.
De finite-, positie- (voor zover ingeschakeld), schaar- en feedbackcontroles blijven
actief. Dit verandert de policy-uitvoering; de opname bewaart de oorspronkelijke
voorspelling en de werkelijk aangevraagde doelen afzonderlijk. Het voorkomt geen
botsingen. Voor een eerste korte proef met nieuwe observaties na elke actie:

```bash
bash scripts/attempt.sh act_rgb begrensd_01 --execute --limit-action-steps --action-count 1 --chunks 1
```

## Poging beëindigen en resultaat markeren

Zorg dat het cameravenster focus heeft:

- `s`: stoppen en resultaat **success** opslaan;
- `f`: stoppen en resultaat **failure** opslaan;
- `q`: stoppen zonder oordeel (**stopped**);
- Ctrl+C in de terminal: **interrupted**.

De toetsen worden verwerkt tussen inferenties en acties; tijdens een lange
inference kan een toets vertraagd reageren. Na stoppen probeert de runner zowel
TCS `halt` als `/scissor/halt` uit te voeren. Een softwaretoets vervangt geen
fysieke stop. Bij een fout wordt **error** opgeslagen met de foutmelding.

## Waar staan de attempts?

```text
runs/attempts/<tijdstip>_<model>_<rgb-of-rgbd>_<naam>/
  attempt.json           resultaat, instellingen, checkpoint, aantal frames
  camera_top.avi         RGB-video bovenste camera
  camera_bottom.avi      RGB-video onderste camera
  observations.jsonl     gemeten J1–J4 en schaarstand, frame-index en tijden
  actions.jsonl          voorspellingen, aangevraagde/afgeronde commando's, tijden
```

De video’s blijven opnemen tijdens inference. Er wordt één cameralezer gedeeld
met de policy; start `record_dataset.py` niet tegelijk op dezelfde camera's.
Video's gebruiken 30 fps als afspeelsnelheid; de werkelijke tijden en gemeten
opnamesnelheid staan in de logs en `attempt.json`. De standen zijn ná elk
camerapaar gemeten, niet zoals demonstraties geïnterpoleerd naar de frametijd.
Deze bestanden zijn evaluatie-opnames, geen automatisch toegevoegde trainingsdata.

Voor RGBD kun je ook ruwe dieptebeelden opslaan:

```bash
bash scripts/attempt.sh act_rgbd poging_01 --execute --record-depth
```

Dit voegt `depth.hdf5` toe en gebruikt meer schijfruimte. Normaal stoppen sluit
video's en dieptebestanden af; bij stroomuitval of geforceerd killen kunnen die
bestanden onvolledig zijn.

## GPU en instellingen

De runner kiest automatisch CUDA als PyTorch een GPU kan gebruiken, anders CPU.
Het gekozen apparaat wordt bij de start getoond. Expliciet afdwingen kan met
`--device cuda:0` of `--device cpu`.

Je gemeten DP RGB CPU-inference duurde ongeveer 5,2 seconden. Daarmee zal
uitvoering stoppen vanwege `max_observation_age_s: 2.0`. Controleer dit vooraf:

```bash
python -c "import torch; print('CUDA beschikbaar:', torch.cuda.is_available())"
python scripts/test_2d_checkpoints.py --variants dp_rgb --device cuda:0
```

Gebruik het tweede commando alleen bij beschikbare CUDA. Een CPU-run die deze
leeftijdscontrole overschrijdt is nog geen bruikbare live-regelkring.

Camera-serienummers, robot-IP, schaarservices en bewegingsinstellingen staan in
`config/robot_2d.yaml`. De oude vaste J1–J4-positiegrenzen staan op jouw verzoek
uit; maximale stappen per commando, finite-controles en schaargrenzen zijn actief.
