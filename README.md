# PFAS Machine Learning Research Repository

## Intro

## Set Up Environment

- Install pyenv
- python version ...

## Notes / Troublshooting

python csv_to_tif.py 

## TODOS

- change dataloader to add in target
- test dataloader on unet

- data augmentation = flips and whatever we do to images
- add readme to processing file folder
- add error checking for lat/lon/target inputs
- for target = specify discrete or boolean
- config pixel size = think more on this
- change the way we handle null vals -50 -> 0 -> 1
- maybe change cell stacking from (x, 1, 10, 10) -> (x, 10, 10)