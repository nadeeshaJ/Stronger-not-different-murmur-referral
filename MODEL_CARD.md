# Model card: CirCor murmur screening models (research use only)

**Not a medical device. Not for clinical use.** These models were built to evaluate referral policies, not to be deployed.

| Item | Details |
|---|---|
| Models | Waveform CNN (587,793 params), spectrogram CNN (293,841), CRNN (509,089); 3 seeds each; 1,125 trained instances across nested cross-validation |
| Input | 4 kHz heart-sound recordings, 20–1000 Hz band-pass, 4 s segments (2 s hop) |
| Output | Encounter-level murmur probability (Platt-calibrated per training split): segment → recording mean → encounter max |
| Training data | CirCor DigiScope v1.0.3: 802 encounters / 744 individuals aged neonate to adolescent, Pernambuco (Brazil) screening campaigns 2014–2015, Littmann 3200 stethoscope; murmur labels from one annotator |
| Intended use | Research on screening and referral evaluation |
| Operating point | Threshold at 90% sensitivity on pooled development out-of-fold scores; outer-test sensitivity 0.90 for all three architectures |
| Performance (outer test, internal) | AUROC 0.908 / 0.916 / 0.935; specificity at the threshold 0.64 / 0.67 / 0.78 (waveform CNN / spectrogram CNN / CRNN) |
| Known weaknesses | Lower sensitivity in adolescents (0.81–0.85); neonates almost absent from training (5 encounters). Under 0 dB noise or dropout, nearly every encounter is flagged (specificity down to 0.09–0.36). Murmur 'Unknown' cases were excluded from training and evaluation. |
| Not validated on | Other populations, devices, sampling rates or clinical settings; external validation is pending (transportability check only) |
| Fairness | Sex and campaign were not analysed as subgroups; age is reported descriptively |
| Contact | Dilani Nadeesha Jayasuriya, Dublin Business School |
