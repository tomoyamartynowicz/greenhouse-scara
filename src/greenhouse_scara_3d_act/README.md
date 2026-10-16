# ACT met pointclouds

De bestaande ACT CVAE/transformer voorspelt 30 absolute vijf-dimensionale acties.
Een DP3 PointNetEncoderXYZRGB verwerkt 1.024 gekleurde punten tot een gepoolde
feature van 64 waarden, die als één visuele token de ACT-transformer ingaat.
De huidige vijf robot-/schaarwaarden en het CVAE-latent vormen de overige input.
Inference gebruikt het gebruikelijke nul-latent; standaard worden de eerste
15 acties uitgevoerd voordat opnieuw wordt waargenomen en voorspeld.

De trainingsloss is L1 over geldige (niet-gepaddedde) targets plus KL met gewicht
10. Normalisatie wordt alleen op trainingsepisodes gefit. De PointNet en
transformer worden samen getraind. Er zijn geen pretrained RGB-gewichten nodig.

Zie [3D_DELFTBLUE.md](../../docs/3D_DELFTBLUE.md) voor pointcloudvoorbereiding,
benaderde kalibratie, Slurm-jobs, checkpoints, offline tests en robotuitvoering.
