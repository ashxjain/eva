## Receptive Field Calculation for Inception v1

![calculations](https://i0.wp.com/syncedreview.com/wp-content/uploads/2017/05/32.png?resize=372%2C171&ssl=1)

|            Type             | Kernel | Stride |  Jin  |  Rin   | Rout              |
| :-------------------------: | :----: | :----: | :---: | :----: | ----------------- |
|         Conv         |  7x7   |   2    |   1   |   1    | 1                 |
|         Max-Pool         |  3x3   |   2    |   2   |   1    | 1+ (3-1)*2 = 5    |
|         Conv         |  3x3   |   1    |   4   |   5    | 5 + (3-1)*4 = 13  |
|         Max-Pool         |  3x3   |   2    |   4   |   13   | 13 + (3-1)*4 = 21 |
|  **Inception(3a): Path1**   |        |        |       |        |                   |
|         Conv         |  1x1   |   1    |   8   |   21   | 21 + (1-1)*8 = 21 |
|  **Inception(3a): Path2**   |        |        |       |        |                   |
|         Conv         |  1x1   |   1    |   8   |   21   | 21 + (1-1)*8 = 21 |
|         Conv         |  3x3   |   1    |   8   |   21   | 21 + (3-1)*8 = 37 |
|  **Inception(3a): Path3**   |        |        |       |        |                   |
|         Conv         |  1x1   |   1    |   8   |   21   | 21 + (1-1)*8 = 21 |
|         Conv         |  5x5   |   1    |   8   |   21   | 21 + (5-1)*8 = 53 |
|  **Inception(3a): Path4**   |        |        |       |        |                   |
|         Max-Pool         |  3x3   |   1    |   8   |   21   | 21 + (3-1)*8 = 37 |
|         Conv         |  1x1   |   1    |   8   |   37   | 37 + (1-1)*8 = 37 |
| **Inception(3a-1): Max RF** | **-**  | **1**  | **8** | **21** | **53**            |

**Since we know that Path3 will yield MaxRF, we will only show that calculation. Also 1x1 doesn't add anything, we will only show 5x5 layer calculation.**

|            Type             | Kernel | Stride |  Jin  |  Rin   | Rout              |
| :------------------------: | :--: | :--: | :--: | :--: | ------------------- |
|        Inc3a-2-Conv         | 5x5  |  1   |  8   |  53  | 53 + (5-1)*8 = 85   |
|        Inc3b-1-Conv         | 5x5  |  1   |  8   |  85  | 85 + (5-1)*8 = 117  |
|        Inc3b-1-Conv         | 5x5  |  1   |  8   | 117  | 117 + (5-1)*8 = 149 |
|        Max-Pool         | 3x3  |  2   | 8 | 149 | 149 + (3-1)*8 = 165 |
|        Inc4a-1-Conv         | 5x5  |  1   |  16  |  165  | 165 + (5-1)*16 = 229 |
|        Inc4a-2-Conv         | 5x5  |  1   |  16  | 229 | 229 + (5-1)*16 = 293 |
|        Inc4b-1-Conv         | 5x5  |  1   |  16  |  293  | 293 + (5-1)*16 = 357 |
|        Inc4b-2-Conv         | 5x5  |  1   |  16  | 357 | 357 + (5-1)*16 = 421 |
|        Inc4c-1-Conv         | 5x5  |  1   |  16  |  421  | 421 + (5-1)*16 = 485 |
|        Inc4c-2-Conv         | 5x5  |  1   |  16  | 485 | 485 + (5-1)*16 = 549 |
|        Inc4d-1-Conv         | 5x5  |  1   |  16  |  549  | 549 + (5-1)*16 = 613 |
|        Inc4d-2-Conv         | 5x5  |  1   |  16  | 613 | 613 + (5-1)*16 = 677 |
|        Inc4e-1-Conv         | 5x5  |  1   |  16  |  677  | 677 + (5-1)*16 = 741 |
|        Inc4e-2-Conv         | 5x5  |  1   |  16  | 741 | 741 + (5-1)*16 = 805 |
| Max-Pool | 3x3 | 2 | 16 | 805 | 805 + (3-1)*16 = 837 |
| Inc5a-1-Conv | 5x5 | 1 | 32 | 837 | 837 + (5-1)*32 = 965 |
| Inc5a-2-Conv | 5x5 | 1 | 32 | 965 | 965 + (5-1)*32 = 1093 |
| Avg-Pool | 7x7 | 1 | 32 | 1093 | 1095 + (7-1)*32 = 1287 |
