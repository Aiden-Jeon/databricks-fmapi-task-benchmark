# n≥1 Repeat Board — mean ± std (M-track)

Each cell = mean ± sample std over that model's valid runs (n). A race is **called** only when the top-two means differ by more than the larger standard error; else **tie**. DNF counts shown when a model failed to submit.

| Task (metric) | Opus 5 | GPT-5.6-sol | GLM 5.2 | verdict |
|---|---|---|---|---|
| t1_pubg (mae) | 0.0572±0.0593 (n=3) | 0.0245±0.0013 (n=3) | 0.1237±0.1246 (n=3) | tie (GPT-5.6-sol≈Opus 5) |
| t2_spooky (multiclass_logloss) | 0.3319±0.0774 (n=3) | 0.3632±0.0130 (n=3) | 0.3843±0.0369 (n=3) | tie (Opus 5≈GPT-5.6-sol) |
| t3_ynat (macro_f1) | 0.8500±0.0012 (n=2) | 0.8402±0.0121 (n=3) | 0.8442±0.0014 (n=2) | **Opus 5** |
| t4_nsmc (accuracy) | 0.8770±0.0067 (n=3) | 0.8767±0.0014 (n=3) | 0.8703±0.0004 (n=2) | tie (Opus 5≈GPT-5.6-sol) |
| t5_bike (rmse) | 316.8700±15.6921 (n=3) | 223.2132±21.7334 (n=3) | 249.0502±48.6038 (n=3) | tie (GPT-5.6-sol≈GLM 5.2) |
| t6_klue_nli (accuracy_str) | 0.8758±0.0034 (n=2) | 0.8745±0.0341 (n=2) | 0.5170±0.0708 (n=3) | tie (Opus 5≈GPT-5.6-sol) |
| t7_klue_sts (pearson) | 0.9586±0.0047 (n=3) | 0.9481±0.0007 (n=3) | 0.9158±0.0446 (n=3) | **Opus 5** |
| t8_beep (macro_f1) | 0.5867±0.0132 (n=3) | 0.5635±0.0017 (n=3) | 0.5511±0.0124 (n=2) | **Opus 5** |
| t9_korquad (korquad) | 0.5159±0.0467 (n=3) | 0.4227±0.0151 (n=2) | 0.0484±0.0000 (n=1)† | **Opus 5** |
| t10_kornli (accuracy_str) | 0.6235±0.0214 (n=3) | 0.6152±0.0189 (n=3) | 0.5278±0.0326 (n=3) | tie (Opus 5≈GPT-5.6-sol) |
| t11_korsts (pearson) | 0.7914±0.0103 (n=3) | 0.7455±0.0083 (n=3) | 0.7534±0.0055 (n=2) | **Opus 5** |
| t12_kobest_boolq (accuracy) | 0.6048±0.0099 (n=3) | 0.6025±0.0044 (n=3) | 0.5748±0.0421 (n=3) | tie (Opus 5≈GPT-5.6-sol) |
| t13_kobest_copa (accuracy) | 0.6391±0.0190 (n=3) | 0.5904±0.0037 (n=3) | 0.6039±0.0184 (n=2) | **Opus 5** |
| t14_kobest_wic (accuracy) | 0.6285±0.0113 (n=3) | 0.5959±0.0087 (n=3) | 0.6182±0.0266 (n=2) | tie (Opus 5≈GLM 5.2) |
| t15_kobest_hellaswag (accuracy) | 0.6786±0.0470 (n=2) | 0.6273±0.0103 (n=3) | 0.5788±0.0309 (n=3) | **Opus 5** |
| t16_kobest_sentineg (accuracy) | 0.9575±0.0024 (n=3) | 0.9539±0.0042 (n=3) | 0.9527±0.0029 (n=2) | **Opus 5** |
| t17_pawsx_ko (accuracy) | 0.7974±0.0438 (n=3) | 0.7602±0.0107 (n=3) | 0.7013±0.0246 (n=3) | **Opus 5** |
| t18_klue_re (accuracy_str) | 0.7713±0.0042 (n=2) | 0.7292±0.0119 (n=3) | 0.7151±0.0082 (n=3) | **Opus 5** |
| t19_klue_mrc (korquad) | 0.3364±0.0139 (n=3) | 0.3375±0.0074 (n=3) | 0.0273±0.0000 (n=1)† | tie (GPT-5.6-sol≈Opus 5) |
| t20_klue_ner (ner_f1) | 0.7860±0.1115 (n=3) | 0.7692±0.0274 (n=3) | 0.7562±0.0191 (n=3) | tie (Opus 5≈GPT-5.6-sol) |
| t21_kmmlu (accuracy) | 0.3394±0.0142 (n=3) | 0.3239±0.0241 (n=3) | 0.3075±0.0000 (n=1)† | **Opus 5** |
| t23_korfin_asc (macro_f1) | 0.7189±0.0208 (n=3) | 0.6757±0.0039 (n=3) | 0.6831±0.0140 (n=2) | **Opus 5** |
| t24_kor_unsmile (multilabel_f1) | 0.7060±0.0383 (n=2) | 0.7140±0.0066 (n=3) | 0.7026±0.0096 (n=2) | tie (GPT-5.6-sol≈Opus 5) |
| t25_klue_dp (las) | 0.7195±0.1297 (n=3) | 0.7674±0.0300 (n=3) | 0.6982±0.0655 (n=3) | tie (GPT-5.6-sol≈Opus 5) |

Decided wins — Opus 5 12 · GPT-5.6-sol 0 · GLM 5.2 0 · statistical ties 12.
† = still n=1 (repeat pending); '?' = leader n=1, undecided.