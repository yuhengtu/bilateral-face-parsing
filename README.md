- Convolutionalize the loss function of Jon Barron’s Fast Bilateral Solver (ECCV 2016 Runner-up) and optimize
depth image with it
- Applying Bilateral Solver as an auxiliary loss function to Face Parsing task to achieve matting

Prepare training data:
- download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)
- change file path in the `prepropess_data.py`  and run

References
- [BiSeNet](https://github.com/CoinCheung/BiSeNet)
- [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
