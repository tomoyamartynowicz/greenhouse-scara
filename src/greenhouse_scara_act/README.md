# Greenhouse SCARA ACT

Aparte versie van `scara_act` met configureerbare RGB/depth-streams en acties die
bij ontbreken tijdens laden uit joints worden afgeleid. De originele map blijft
intact. Zie ook [repository- en clusterinstructies](../../README.md). Zie [gedeelde uitleg](../greenhouse_scara_common/README.md) voor het
inputformaat, chunks, depth en tests.

## Inputs

Pas `inputs.yaml` aan. Standaard: RGB en depth van `camera_top`, plus joints.
Extra camera's toevoegen of depth uitschakelen kan met de entries in `shape_meta`.
Alle ACT-streams moeten dezelfde doelresolutie hebben.

Alle RGB-camera's delen één backbone; alle depth-camera's delen een tweede.
Twee RGB-D-camera's gebruiken dus twee backbones. De backbonegewichten worden
meegetraind; alleen BatchNorm is frozen, zoals in de originele ACT-code.
De beelden blijven een dictionary omdat RGB drie kanalen heeft en depth twee:
de afstand en een geldigheidsmasker (0 bij ontbrekende meting, anders 1).
Met `depth_normalization_m: 1.0` is de afstand in meters, zonder clipping.
Een waarde boven 1 blijft behouden. Training en live-inference gebruiken dezelfde
verwerking. Oude configuraties met `max_depth_m` behouden hun eerdere clipping.

Voor kinesthetic teaching wordt zonder opgeslagen `/action` het doel op tijdstip
`t` afgeleid als `qpos[t+1]` (`action_offset: 1`). Aan het episode-einde wordt de
laatste positie herhaald. Dit is een volgend bewegingsdoel, geen compensatie voor
vertraging. Als `/action` bestaat, wordt bij `action_source: auto` direct
`action[t]` gebruikt, zonder de `t-1`-verschuiving uit originele ACT.
De dataset kiest net als ACT een willekeurig tijdstip per episode, maar laadt
direct de benodigde chunk in plaats van eerst de hele resterende episode.

## Trainen

Vanuit de repositoryroot `greenhouse-scara`, met de bestaande `act`-omgeving:

```bash
python src/greenhouse_scara_act/imitate_episodes.py \
  --dataset_dir /pad/naar/greenhouse_demos \
  --ckpt_dir runs/greenhouse_scara_act/rgbd_run01 \
  --inputs_config src/greenhouse_scara_act/inputs.yaml \
  --batch_size 8 --chunk_size 30 --num_epochs 2000
```

Een korte CPU-controle op de voorbeeldopname, zonder acties vooraf te maken:

```bash
OMP_NUM_THREADS=1 python src/greenhouse_scara_act/imitate_episodes.py \
  --dataset_dir datasets/test_rgbd_single_joints \
  --ckpt_dir runs/greenhouse_scara_act/smoke01 \
  --allow_single_episode --device cpu --num_workers 0 \
  --no-pretrained_backbone --batch_size 1 --num_epochs 1 \
  --chunk_size 4 --hidden_dim 32 --dim_feedforward 64 \
  --max_train_steps 1 --max_val_steps 1
```

De controle gebruikt een kleinere transformer en dezelfde demo voor training en
validatie. Gebruik die resultaten niet als kwaliteitsmeting. Voor normale training
gebruikt ACT standaard pretrained RGB-backbones; depth krijgt een aangepaste
inputconvolutie. Met `--no-pretrained_backbone` worden geen pretrained gewichten
opgehaald. Normalisatie voor RGB en depth is gescheiden.

De runmap bevat `config.pkl`, `dataset_stats.pkl`, `policy_best.ckpt`,
`policy_last.ckpt` en periodieke checkpoints. Gebruik een nieuwe runmap voor een
andere streamconfiguratie. Het gekopieerde `imitate_episodes_resume.py` kan ook
met `--inputs_config` worden gebruikt; hervatten vereist zijn eigen
`training_state.ckpt` en exact dezelfde inputconfiguratie.
Begin ook een nieuwe training bij overstappen van aparte backbones per stream
naar gedeelde backbones, of van geclipte diepte naar diepte in meters.
Oude checkpoints met meerdere camera's per modaliteit passen niet rechtstreeks
op de nieuwe backbone-indeling.

## Live-inference

```bash
python src/greenhouse_scara_act/eval.py \
  --ckpt-dir runs/greenhouse_scara_act/rgbd_run01 \
  --checkpoint policy_best.ckpt --chunks 1 --no-viewer
```

Standaard geen robotbeweging. De streams komen uit het opgeslagen configuratiebestand;
`--camera NAME SERIAL` kan de fysieke camera's aangeven. De aparte
`--execute`-optie uit de oorspronkelijke evaluatie schakelt beweging in.
Het live RGB-venster blijft een preview; de policy ontvangt ook de geconfigureerde depth.

De ongebruikte oude `camera.py`, het hardcoded afspeelscript `test.py` en de
legacy `visualize_episodes.py` zijn verwijderd. Live-invoer gebruikt
`greenhouse_scara_common`; datasetweergave kan met de notebook in `scara_dataset`.
De ACT-modelcode, training, hervatten en robotviewer zijn behouden.
