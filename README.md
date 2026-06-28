
# subphot
`subphot` stands for Subtraction Photometry!

`subphot` is a simple difference photometry pipeline using [`hotpants`](https://github.com/acbecker/hotpants). I made this for specifically LCO images.

This pipeline does not support .fits.fz files.

#### Code to run the pipeline
```
python diff_phot_par.py --sci "/home/joysankar/LCO/IP_IMAGES" --ref "/home/joysankar/LCO/ref" --ra 90.93754 --dec -64.37665 --outdir "/home/joysankar/LCO/diff_i" --hotpants "/home/joysankar/hotpants/hotpants" --ncpu 8
```

#### The packages you need to install 
- astropy
- photutils
- astroquery
- setup hotpants