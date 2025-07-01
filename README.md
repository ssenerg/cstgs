# Counting and Sampling Triangles from a Graph Stream

## Algorithm's Scientific Paper

- https://www.vldb.org/pvldb/vol6/p1870-aduri.pdf
---

## Results 

### 64 Processors

```
┌─────────────────────────────────┬───────────┬──────────────┐
│ dataset                         ┆ triangles ┆ process_took │
│ ---                             ┆ ---       ┆ ---          │
│ str                             ┆ i64       ┆ str          │
╞═════════════════════════════════╪═══════════╪══════════════╡
│ Oregon-1::oregon1_010421.tsv    ┆ 56        ┆ 33.60ms      │
│ Oregon-1::oregon1_010505.tsv    ┆ 48        ┆ 43.30ms      │
│ Oregon-1::oregon1_010526.tsv    ┆ 57        ┆ 64.98ms      │
│ Oregon-1::oregon1_010407.tsv    ┆ 29        ┆ 65.96ms      │
│ Oregon-1::oregon1_010428.tsv    ┆ 36        ┆ 76.21ms      │
│ Oregon-1::oregon1_010331.tsv    ┆ 47        ┆ 78.15ms      │
│ Oregon-1::oregon1_010414.tsv    ┆ 56        ┆ 78.56ms      │
│ Oregon-1::oregon1_010512.tsv    ┆ 48        ┆ 80.28ms      │
│ Oregon-1::oregon1_010519.tsv    ┆ 39        ┆ 80.74ms      │
│ Caida::as-caida20070917.tsv     ┆ 65        ┆ 84.65ms      │
│ Caida::as-caida20040202.tsv     ┆ 111       ┆ 126.77ms     │
│ Caida::as-caida20040503.tsv     ┆ 114       ┆ 133.84ms     │
│ Caida::as-caida20040607.tsv     ┆ 111       ┆ 138.63ms     │
│ Reddit::soc-redditHyperlinks-b… ┆ 17460     ┆ 145.43ms     │
│ Caida::as-caida20060313.tsv     ┆ 138       ┆ 150.39ms     │
│ Caida::as-caida20050704.tsv     ┆ 130       ┆ 150.56ms     │
│ Caida::as-caida20050207.tsv     ┆ 81        ┆ 151.37ms     │
│ Caida::as-caida20070716.tsv     ┆ 176       ┆ 152.70ms     │
│ Caida::as-caida20040405.tsv     ┆ 104       ┆ 161.43ms     │
│ Caida::as-caida20040105.tsv     ┆ 58        ┆ 164.59ms     │
│ Caida::as-caida20070212.tsv     ┆ 220       ┆ 175.15ms     │
│ Caida::as-caida20060116.tsv     ┆ 148       ┆ 182.92ms     │
│ Caida::as-caida20050307.tsv     ┆ 81        ┆ 187.89ms     │
│ Caida::as-caida20060508.tsv     ┆ 207       ┆ 191.24ms     │
│ Caida::as-caida20040301.tsv     ┆ 70        ┆ 194.06ms     │
│ Caida::as-caida20050404.tsv     ┆ 120       ┆ 195.29ms     │
│ Caida::as-caida20060327.tsv     ┆ 182       ┆ 196.45ms     │
│ Caida::as-caida20060403.tsv     ┆ 189       ┆ 196.47ms     │
│ Caida::as-caida20050801.tsv     ┆ 119       ┆ 196.62ms     │
│ Caida::as-caida20040906.tsv     ┆ 195       ┆ 197.47ms     │
│ Caida::as-caida20040802.tsv     ┆ 89        ┆ 202.06ms     │
│ Caida::as-caida20061211.tsv     ┆ 163       ┆ 202.38ms     │
│ Caida::as-caida20070108.tsv     ┆ 156       ┆ 205.16ms     │
│ Caida::as-caida20041101.tsv     ┆ 123       ┆ 205.85ms     │
│ Caida::as-caida20061016.tsv     ┆ 164       ┆ 212.41ms     │
│ Caida::as-caida20060220.tsv     ┆ 153       ┆ 213.78ms     │
│ Caida::as-caida20060710.tsv     ┆ 163       ┆ 214.82ms     │
│ Caida::as-caida20060501.tsv     ┆ 215       ┆ 217.63ms     │
│ Caida::as-caida20070129.tsv     ┆ 190       ┆ 218.62ms     │
│ Caida::as-caida20051107.tsv     ┆ 158       ┆ 223.62ms     │
│ Caida::as-caida20050606.tsv     ┆ 114       ┆ 227.72ms     │
│ Caida::as-caida20060227.tsv     ┆ 158       ┆ 228.09ms     │
│ Caida::as-caida20060731.tsv     ┆ 113       ┆ 234.79ms     │
│ Caida::as-caida20060206.tsv     ┆ 130       ┆ 241.67ms     │
│ Caida::as-caida20060717.tsv     ┆ 145       ┆ 242.11ms     │
│ Caida::as-caida20050103.tsv     ┆ 82        ┆ 243.10ms     │
│ Caida::as-caida20060123.tsv     ┆ 72        ┆ 247.41ms     │
│ Caida::as-caida20061113.tsv     ┆ 22        ┆ 248.08ms     │
│ Caida::as-caida20041206.tsv     ┆ 130       ┆ 248.53ms     │
│ Caida::as-caida20060522.tsv     ┆ 218       ┆ 249.20ms     │
│ Caida::as-caida20060410.tsv     ┆ 106       ┆ 249.89ms     │
│ Caida::as-caida20060130.tsv     ┆ 160       ┆ 250.17ms     │
│ Caida::as-caida20060306.tsv     ┆ 168       ┆ 251.48ms     │
│ Caida::as-caida20070813.tsv     ┆ 125       ┆ 253.41ms     │
│ Caida::as-caida20051003.tsv     ┆ 111       ┆ 255.60ms     │
│ Caida::as-caida20060724.tsv     ┆ 187       ┆ 258.14ms     │
│ Caida::as-caida20060814.tsv     ┆ 177       ┆ 260.00ms     │
│ Caida::as-caida20060109.tsv     ┆ 120       ┆ 260.27ms     │
│ Caida::as-caida20070521.tsv     ┆ 148       ┆ 261.80ms     │
│ Caida::as-caida20060807.tsv     ┆ 135       ┆ 262.72ms     │
│ Caida::as-caida20061002.tsv     ┆ 147       ┆ 264.33ms     │
│ Caida::as-caida20070115.tsv     ┆ 156       ┆ 266.50ms     │
│ Caida::as-caida20060102.tsv     ┆ 107       ┆ 270.38ms     │
│ Caida::as-caida20061225.tsv     ┆ 204       ┆ 276.06ms     │
│ Caida::as-caida20070219.tsv     ┆ 138       ┆ 276.92ms     │
│ Caida::as-caida20070409.tsv     ┆ 187       ┆ 283.73ms     │
│ Caida::as-caida20070416.tsv     ┆ 187       ┆ 284.09ms     │
│ Caida::as-caida20061030.tsv     ┆ 186       ┆ 285.96ms     │
│ Caida::as-caida20041004.tsv     ┆ 144       ┆ 287.17ms     │
│ Caida::as-caida20060529.tsv     ┆ 122       ┆ 287.51ms     │
│ Caida::as-caida20040705.tsv     ┆ 102       ┆ 288.46ms     │
│ Caida::as-caida20061106.tsv     ┆ 99        ┆ 288.62ms     │
│ Caida::as-caida20060213.tsv     ┆ 148       ┆ 292.63ms     │
│ Caida::as-caida20060703.tsv     ┆ 154       ┆ 295.57ms     │
│ Caida::as-caida20070122.tsv     ┆ 177       ┆ 296.96ms     │
│ Caida::as-caida20070514.tsv     ┆ 229       ┆ 297.73ms     │
│ Caida::as-caida20060605.tsv     ┆ 217       ┆ 298.77ms     │
│ Caida::as-caida20060821.tsv     ┆ 177       ┆ 301.13ms     │
│ Caida::as-caida20050905.tsv     ┆ 118       ┆ 302.73ms     │
│ Caida::as-caida20060619.tsv     ┆ 179       ┆ 304.70ms     │
│ Caida::as-caida20061023.tsv     ┆ 160       ┆ 305.06ms     │
│ Caida::as-caida20070709.tsv     ┆ 125       ┆ 305.69ms     │
│ Caida::as-caida20071022.tsv     ┆ 176       ┆ 306.48ms     │
│ Caida::as-caida20060918.tsv     ┆ 219       ┆ 306.86ms     │
│ Caida::as-caida20070618.tsv     ┆ 145       ┆ 308.67ms     │
│ Caida::as-caida20070604.tsv     ┆ 163       ┆ 309.57ms     │
│ Caida::as-caida20070820.tsv     ┆ 156       ┆ 309.80ms     │
│ Caida::as-caida20060612.tsv     ┆ 198       ┆ 310.04ms     │
│ Caida::as-caida20060925.tsv     ┆ 189       ┆ 310.77ms     │
│ Caida::as-caida20060626.tsv     ┆ 173       ┆ 311.92ms     │
│ Caida::as-caida20071001.tsv     ┆ 153       ┆ 313.25ms     │
│ Caida::as-caida20070312.tsv     ┆ 178       ┆ 314.70ms     │
│ Caida::as-caida20070806.tsv     ┆ 172       ┆ 318.49ms     │
│ Caida::as-caida20071105.tsv     ┆ 177       ┆ 319.85ms     │
│ Caida::as-caida20060424.tsv     ┆ 184       ┆ 320.42ms     │
│ Caida::as-caida20070528.tsv     ┆ 140       ┆ 323.31ms     │
│ Caida::as-caida20070430.tsv     ┆ 216       ┆ 324.31ms     │
│ Caida::as-caida20061204.tsv     ┆ 145       ┆ 325.98ms     │
│ Caida::as-caida20070305.tsv     ┆ 161       ┆ 327.21ms     │
│ Caida::as-caida20070730.tsv     ┆ 171       ┆ 330.43ms     │
│ Caida::as-caida20060320.tsv     ┆ 136       ┆ 330.78ms     │
│ Caida::as-caida20070205.tsv     ┆ 156       ┆ 331.79ms     │
│ Caida::as-caida20060911.tsv     ┆ 154       ┆ 335.42ms     │
│ Reddit::soc-redditHyperlinks-t… ┆ 53501     ┆ 336.73ms     │
│ Caida::as-caida20060828.tsv     ┆ 179       ┆ 336.99ms     │
│ Caida::as-caida20060515.tsv     ┆ 192       ┆ 337.85ms     │
│ Caida::as-caida20070101.tsv     ┆ 174       ┆ 338.84ms     │
│ Caida::as-caida20061009.tsv     ┆ 186       ┆ 339.46ms     │
│ Caida::as-caida20071029.tsv     ┆ 191       ┆ 344.95ms     │
│ Caida::as-caida20050502.tsv     ┆ 146       ┆ 348.23ms     │
│ Caida::as-caida20070402.tsv     ┆ 180       ┆ 349.87ms     │
│ Caida::as-caida20060417.tsv     ┆ 168       ┆ 351.40ms     │
│ Caida::as-caida20071015.tsv     ┆ 152       ┆ 352.32ms     │
│ Caida::as-caida20051205.tsv     ┆ 163       ┆ 353.05ms     │
│ Caida::as-caida20070611.tsv     ┆ 175       ┆ 354.28ms     │
│ Caida::as-caida20070827.tsv     ┆ 180       ┆ 355.60ms     │
│ Caida::as-caida20061120.tsv     ┆ 144       ┆ 365.43ms     │
│ Caida::as-caida20061127.tsv     ┆ 176       ┆ 370.42ms     │
│ Caida::as-caida20070319.tsv     ┆ 203       ┆ 371.66ms     │
│ Caida::as-caida20070507.tsv     ┆ 170       ┆ 371.91ms     │
│ Caida::as-caida20071008.tsv     ┆ 173       ┆ 372.96ms     │
│ Caida::as-caida20071112.tsv     ┆ 130       ┆ 376.21ms     │
│ Caida::as-caida20070226.tsv     ┆ 173       ┆ 376.60ms     │
│ Caida::as-caida20070702.tsv     ┆ 176       ┆ 376.61ms     │
│ Caida::as-caida20060904.tsv     ┆ 174       ┆ 381.28ms     │
│ Caida::as-caida20070924.tsv     ┆ 160       ┆ 387.30ms     │
│ Caida::as-caida20070910.tsv     ┆ 181       ┆ 389.14ms     │
│ Caida::as-caida20070723.tsv     ┆ 193       ┆ 390.89ms     │
│ Caida::as-caida20070625.tsv     ┆ 142       ┆ 413.54ms     │
│ Caida::as-caida20061218.tsv     ┆ 131       ┆ 433.23ms     │
│ Caida::as-caida20070423.tsv     ┆ 160       ┆ 436.45ms     │
│ Caida::as-caida20070903.tsv     ┆ 186       ┆ 456.83ms     │
│ Caida::as-caida20070326.tsv     ┆ 178       ┆ 461.07ms     │
└─────────────────────────────────┴───────────┴──────────────┘ 
```

### 12 Processors

```
┌─────────────────────────────────┬───────────┬──────────────┐
│ dataset                         ┆ triangles ┆ process_took │
│ ---                             ┆ ---       ┆ ---          │
│ str                             ┆ i64       ┆ str          │
╞═════════════════════════════════╪═══════════╪══════════════╡
│ Oregon-1::oregon1_010421.tsv    ┆ 347       ┆ 82.91ms      │
│ Oregon-1::oregon1_010526.tsv    ┆ 468       ┆ 127.04ms     │
│ Oregon-1::oregon1_010505.tsv    ┆ 424       ┆ 137.70ms     │
│ Oregon-1::oregon1_010512.tsv    ┆ 433       ┆ 143.06ms     │
│ Oregon-1::oregon1_010519.tsv    ┆ 335       ┆ 153.31ms     │
│ Oregon-1::oregon1_010407.tsv    ┆ 431       ┆ 155.90ms     │
│ Oregon-1::oregon1_010414.tsv    ┆ 359       ┆ 156.39ms     │
│ Oregon-1::oregon1_010331.tsv    ┆ 519       ┆ 165.71ms     │
│ Oregon-1::oregon1_010428.tsv    ┆ 450       ┆ 166.81ms     │
│ Caida::as-caida20070917.tsv     ┆ 2114      ┆ 168.76ms     │
│ Caida::as-caida20060109.tsv     ┆ 2899      ┆ 267.86ms     │
│ Caida::as-caida20050606.tsv     ┆ 2993      ┆ 442.04ms     │
│ Caida::as-caida20040802.tsv     ┆ 2647      ┆ 452.76ms     │
│ Caida::as-caida20070416.tsv     ┆ 4417      ┆ 456.99ms     │
│ Caida::as-caida20060424.tsv     ┆ 3148      ┆ 468.95ms     │
│ Caida::as-caida20060724.tsv     ┆ 3680      ┆ 483.07ms     │
│ Reddit::soc-redditHyperlinks-b… ┆ 406010    ┆ 484.48ms     │
│ Caida::as-caida20070319.tsv     ┆ 4729      ┆ 489.45ms     │
│ Caida::as-caida20060403.tsv     ┆ 3344      ┆ 544.35ms     │
│ Caida::as-caida20061002.tsv     ┆ 3558      ┆ 545.61ms     │
│ Caida::as-caida20050801.tsv     ┆ 2987      ┆ 548.43ms     │
│ Caida::as-caida20070326.tsv     ┆ 4377      ┆ 550.61ms     │
│ Caida::as-caida20050905.tsv     ┆ 3051      ┆ 551.48ms     │
│ Caida::as-caida20070910.tsv     ┆ 3956      ┆ 554.11ms     │
│ Caida::as-caida20060612.tsv     ┆ 3171      ┆ 557.91ms     │
│ Caida::as-caida20051205.tsv     ┆ 2835      ┆ 576.78ms     │
│ Caida::as-caida20070521.tsv     ┆ 3873      ┆ 578.24ms     │
│ Caida::as-caida20060731.tsv     ┆ 3207      ┆ 579.16ms     │
│ Caida::as-caida20070723.tsv     ┆ 4118      ┆ 581.67ms     │
│ Caida::as-caida20060508.tsv     ┆ 3287      ┆ 583.11ms     │
│ Caida::as-caida20060320.tsv     ┆ 3723      ┆ 588.74ms     │
│ Caida::as-caida20070108.tsv     ┆ 4275      ┆ 597.21ms     │
│ Caida::as-caida20041004.tsv     ┆ 3200      ┆ 600.79ms     │
│ Caida::as-caida20070129.tsv     ┆ 4198      ┆ 613.91ms     │
│ Caida::as-caida20060313.tsv     ┆ 3474      ┆ 617.63ms     │
│ Caida::as-caida20060227.tsv     ┆ 3500      ┆ 621.12ms     │
│ Caida::as-caida20041206.tsv     ┆ 2998      ┆ 622.47ms     │
│ Caida::as-caida20040607.tsv     ┆ 2911      ┆ 623.27ms     │
│ Caida::as-caida20040202.tsv     ┆ 2508      ┆ 626.48ms     │
│ Caida::as-caida20040405.tsv     ┆ 2700      ┆ 627.15ms     │
│ Caida::as-caida20060710.tsv     ┆ 3112      ┆ 632.68ms     │
│ Caida::as-caida20070219.tsv     ┆ 3649      ┆ 638.75ms     │
│ Caida::as-caida20060529.tsv     ┆ 3408      ┆ 639.40ms     │
│ Caida::as-caida20041101.tsv     ┆ 3175      ┆ 640.89ms     │
│ Caida::as-caida20061016.tsv     ┆ 3624      ┆ 642.85ms     │
│ Caida::as-caida20060522.tsv     ┆ 3215      ┆ 647.13ms     │
│ Caida::as-caida20061218.tsv     ┆ 3891      ┆ 650.60ms     │
│ Caida::as-caida20061113.tsv     ┆ 942       ┆ 655.03ms     │
│ Caida::as-caida20070423.tsv     ┆ 3833      ┆ 655.25ms     │
│ Caida::as-caida20060703.tsv     ┆ 3199      ┆ 662.82ms     │
│ Caida::as-caida20050307.tsv     ┆ 2947      ┆ 666.75ms     │
│ Caida::as-caida20060220.tsv     ┆ 3651      ┆ 672.38ms     │
│ Caida::as-caida20060213.tsv     ┆ 3318      ┆ 673.93ms     │
│ Caida::as-caida20060130.tsv     ┆ 3415      ┆ 676.25ms     │
│ Caida::as-caida20060821.tsv     ┆ 3490      ┆ 676.72ms     │
│ Caida::as-caida20050502.tsv     ┆ 2992      ┆ 682.27ms     │
│ Caida::as-caida20060306.tsv     ┆ 3445      ┆ 686.63ms     │
│ Caida::as-caida20060918.tsv     ┆ 3605      ┆ 690.70ms     │
│ Caida::as-caida20060807.tsv     ┆ 3591      ┆ 690.79ms     │
│ Caida::as-caida20070212.tsv     ┆ 4180      ┆ 696.60ms     │
│ Caida::as-caida20060206.tsv     ┆ 3241      ┆ 697.30ms     │
│ Caida::as-caida20050704.tsv     ┆ 3125      ┆ 698.84ms     │
│ Caida::as-caida20050404.tsv     ┆ 2732      ┆ 702.49ms     │
│ Caida::as-caida20040906.tsv     ┆ 3305      ┆ 703.50ms     │
│ Caida::as-caida20040705.tsv     ┆ 2679      ┆ 705.90ms     │
│ Caida::as-caida20050103.tsv     ┆ 2942      ┆ 713.95ms     │
│ Caida::as-caida20060123.tsv     ┆ 2932      ┆ 716.87ms     │
│ Caida::as-caida20070305.tsv     ┆ 3845      ┆ 719.71ms     │
│ Caida::as-caida20060102.tsv     ┆ 3004      ┆ 720.61ms     │
│ Caida::as-caida20061023.tsv     ┆ 3743      ┆ 732.11ms     │
│ Caida::as-caida20060605.tsv     ┆ 3367      ┆ 734.65ms     │
│ Caida::as-caida20061009.tsv     ┆ 4038      ┆ 737.31ms     │
│ Caida::as-caida20040503.tsv     ┆ 2744      ┆ 742.23ms     │
│ Caida::as-caida20061030.tsv     ┆ 3006      ┆ 743.86ms     │
│ Caida::as-caida20061211.tsv     ┆ 4455      ┆ 743.99ms     │
│ Caida::as-caida20061120.tsv     ┆ 3889      ┆ 749.24ms     │
│ Caida::as-caida20060501.tsv     ┆ 3645      ┆ 751.47ms     │
│ Caida::as-caida20060417.tsv     ┆ 3433      ┆ 757.10ms     │
│ Caida::as-caida20060327.tsv     ┆ 3510      ┆ 759.49ms     │
│ Caida::as-caida20070402.tsv     ┆ 4133      ┆ 763.83ms     │
│ Caida::as-caida20060904.tsv     ┆ 3977      ┆ 773.27ms     │
│ Caida::as-caida20070820.tsv     ┆ 4023      ┆ 775.34ms     │
│ Caida::as-caida20060814.tsv     ┆ 3154      ┆ 783.09ms     │
│ Caida::as-caida20070618.tsv     ┆ 4016      ┆ 789.86ms     │
│ Caida::as-caida20060619.tsv     ┆ 3452      ┆ 800.58ms     │
│ Caida::as-caida20070312.tsv     ┆ 4359      ┆ 804.32ms     │
│ Caida::as-caida20071112.tsv     ┆ 3228      ┆ 809.96ms     │
│ Caida::as-caida20070205.tsv     ┆ 3820      ┆ 813.28ms     │
│ Caida::as-caida20050207.tsv     ┆ 2503      ┆ 814.21ms     │
│ Caida::as-caida20051107.tsv     ┆ 2856      ┆ 817.95ms     │
│ Caida::as-caida20070226.tsv     ┆ 4004      ┆ 820.58ms     │
│ Caida::as-caida20060828.tsv     ┆ 3543      ┆ 822.81ms     │
│ Caida::as-caida20060911.tsv     ┆ 4216      ┆ 836.62ms     │
│ Caida::as-caida20070903.tsv     ┆ 3528      ┆ 837.86ms     │
│ Caida::as-caida20070625.tsv     ┆ 4272      ┆ 839.07ms     │
│ Caida::as-caida20071015.tsv     ┆ 4159      ┆ 841.44ms     │
│ Caida::as-caida20070514.tsv     ┆ 4177      ┆ 844.88ms     │
│ Caida::as-caida20040301.tsv     ┆ 2311      ┆ 852.71ms     │
│ Caida::as-caida20061204.tsv     ┆ 4007      ┆ 853.92ms     │
│ Caida::as-caida20070507.tsv     ┆ 4217      ┆ 865.13ms     │
│ Caida::as-caida20070827.tsv     ┆ 4095      ┆ 867.37ms     │
│ Caida::as-caida20060626.tsv     ┆ 3338      ┆ 867.79ms     │
│ Caida::as-caida20051003.tsv     ┆ 2735      ┆ 871.30ms     │
│ Caida::as-caida20070528.tsv     ┆ 3814      ┆ 881.36ms     │
│ Caida::as-caida20070730.tsv     ┆ 4346      ┆ 888.35ms     │
│ Caida::as-caida20060116.tsv     ┆ 3171      ┆ 903.64ms     │
│ Caida::as-caida20070409.tsv     ┆ 4077      ┆ 921.09ms     │
│ Caida::as-caida20070813.tsv     ┆ 3429      ┆ 929.55ms     │
│ Caida::as-caida20060515.tsv     ┆ 3391      ┆ 943.29ms     │
│ Caida::as-caida20061127.tsv     ┆ 3873      ┆ 946.83ms     │
│ Caida::as-caida20071029.tsv     ┆ 4022      ┆ 947.07ms     │
│ Caida::as-caida20071105.tsv     ┆ 3680      ┆ 951.38ms     │
│ Caida::as-caida20071001.tsv     ┆ 3789      ┆ 965.24ms     │
│ Caida::as-caida20070702.tsv     ┆ 4569      ┆ 970.16ms     │
│ Caida::as-caida20070430.tsv     ┆ 4407      ┆ 971.33ms     │
│ Caida::as-caida20070115.tsv     ┆ 4320      ┆ 976.59ms     │
│ Caida::as-caida20070101.tsv     ┆ 4305      ┆ 986.08ms     │
│ Caida::as-caida20070709.tsv     ┆ 4453      ┆ 992.13ms     │
│ Caida::as-caida20070604.tsv     ┆ 4372      ┆ 1.01s        │
│ Caida::as-caida20070122.tsv     ┆ 4406      ┆ 1.01s        │
│ Caida::as-caida20070716.tsv     ┆ 4435      ┆ 1.02s        │
│ Caida::as-caida20060717.tsv     ┆ 3287      ┆ 1.03s        │
│ Caida::as-caida20061106.tsv     ┆ 3138      ┆ 1.03s        │
│ Caida::as-caida20060925.tsv     ┆ 3906      ┆ 1.03s        │
│ Caida::as-caida20070924.tsv     ┆ 4129      ┆ 1.03s        │
│ Caida::as-caida20060410.tsv     ┆ 2887      ┆ 1.03s        │
│ Caida::as-caida20040105.tsv     ┆ 2452      ┆ 1.06s        │
│ Caida::as-caida20070806.tsv     ┆ 4176      ┆ 1.08s        │
│ Caida::as-caida20071008.tsv     ┆ 3896      ┆ 1.08s        │
│ Caida::as-caida20070611.tsv     ┆ 3793      ┆ 1.10s        │
│ Caida::as-caida20061225.tsv     ┆ 4435      ┆ 1.22s        │
│ Caida::as-caida20071022.tsv     ┆ 3745      ┆ 1.28s        │
│ Reddit::soc-redditHyperlinks-t… ┆ 2368752   ┆ 2.34s        │
└─────────────────────────────────┴───────────┴──────────────┘
```

### 2 Processors - Best Estimation

```

┌─────────────────────────────────┬───────────┬──────────────────┬──────────────┐
│ dataset                         ┆ triangles ┆ actual_triangles ┆ process_took │
│ ---      -e = 0.8 , -w = 1.0    ┆ ---       ┆ ---              ┆ ---          │
│ str                             ┆ i64       ┆ i64              ┆ str          │
╞═════════════════════════════════╪═══════════╪══════════════════╪══════════════╡
│ Oregon-1::oregon1_010331.tsv    ┆ 5679      ┆ 17144            ┆ 671.30ms     │
│ Caida::as-caida20060102.tsv     ┆ 39288     ┆ 30433            ┆ 4.11s        │
│ Caida::as-caida20070101.tsv     ┆ 52210     ┆ 40475            ┆ 5.40s        │
│ Reddit::soc-redditHyperlinks-b… ┆ 5800025   ┆ 406391           ┆ 16.55s       │
└─────────────────────────────────┴───────────┴──────────────────┴──────────────┘
```
---

### Notes

- Increasing processor numbers means **less triangles count** and **less runtime**.
- Increasing edge and wedge sampling probability means **more triangles count** and **more runtime** .
---

### Used data is accessible via these links

```
https://snap.stanford.edu/data/as-Caida.html
https://snap.stanford.edu/data/soc-RedditHyperlinks.html
https://snap.stanford.edu/data/Oregon-1.html
```