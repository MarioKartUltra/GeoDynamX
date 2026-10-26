# GeoDynamix (beta)

[![tests](https://github.com/MarioKartUltra/GeoDynamX/actions/workflows/tests.yml/badge.svg)](https://github.com/MarioKartUltra/GeoDynamX/actions/workflows/tests.yml)

GeoDynamix is an application for mapping and analysis of rasters. It started as a dedicated GUI for fast and efficient multiscale analysis of geoscientific rasters. I (Abe) particularly made the application for metamorphic geology (see Hobbs and Ord 2015). Metamorphic geology is far more generalized than metamorphic petrology, so a lot of the concepts translate to any other branch of geoscience involving multiple scales in time and or space. GeoDynamX can be especially useful in metamorphic geology, geomorphology, and tectonic geomorphology. Although if geochemists ever become interesting in asking how geochemistry is spatially organized (hint hint: what's the r**2 in your correlations and is it so low becasue spatial information is destroyed?). 

More generally, this application can be used in any discipline concerned with multiscale analysis and edge detection. 


GeoDynamX allows one to quickyl measure how structure changes with scale, using wavelet-based multifractal methods, edge detection, matrix decompositions, etc. and takes care of georeferencing and processing provenance on-the-fly. The 3D globe view is currently broken and is not really a priority, so if anyone wants to work on that, go ahead. 



Each dataset is analyzed on its own native grid such that the results can be projected, because
resampling a raster before a scale-sensitive analysis changes the statistics being measured. In some cases some level of reprojection is necessary, and in this case we can also back-project the information. 


GeoDynamX uses Abe's GeoProvenance (.gprv) metadata framework, which is in development. The goal is to track every processing step a way that enables reproducible science. Since there is a lot of hyperparameter tuning involved with some tools in nonlinear dynamics, the author had a simple idea: In geology, it is acceptable to "interpret" features on screenshots in Adobe Illustrator. Therefore, a geologist can tune hyperparameters until features of interest show up rather than depending on strict mathematical methods for tuning the analyses. Of course one can still do that. The goal of teh GPRV file is that when a user exports their final product, a full tree of instructions comes with it that allows another user to import the raw data (or at least the initial  state of teh data used) and process it through the same workflow with all of the hyperparameters set where they need to be to reproduce the data product. Hopefully this type of work can inspire more open data use across geoscientists. While some groups are good about metadata, others have poor records and only give descriptions about processing in methods sections of papers. Moreover, some scientists like to LARP about enabling open science by sharing their data, then they do not share any data, only the processed products that they used in their papers. Perhaps this enables reproducible statistics, but does not enable reproducible science, and the only thing it really "enables" is for the person who processed their own data to be cited for their product instead of being cited for their data. 

A gprv record is canonical JSON (RFC 8785)  and is keyed by ISO 19115-3 lineage names. This means the fields are the ISO standard's own lineage elements, and one can, in theory, export a GPRV file to ISO standard (which is what programs like ARCGis use; however you amy find the provenance or metadata to be empty when a geoscientist shares geospatial information). One step was mechanically exported to 19115-3 XML and validated against the official lineage schema (mrl/2.0, and mrl/1.0) by checking with the 19115-3 XSD bundle. ISO 19157 (data quality) is noted as a possible later export. The GPRV framework is still in development and I'm not sure if it's really working yet (not my highest priority yet). 

## What is available in GeoDynamX?
- **A quick notes about wavelets and "multifractal" / microlocal analysis** Many implementations of wavelet transform default to the engineering convention where the dilation norm is L2. At a local singularity , this will shift the holder exponent by 1/2, although I think it wouldn't change the results from canonical partition functions (not sure about this). For this reason, the default scaling for all wavelet transforms is an L1 norm. 

 Multifractal analysis was, scientifically,  first used to quantify the dynamics of turbulent flows. This was done by covering an attractor with boxes and extracting they Reyni dimensions. This method was difficult to apply to fully developed turbulence becasue the embedding dimension would be far too high. Frisch and Parisi, I believe, were the first to develop an incrimental approach, sparking the use of multifractal formalisms for functions (rather than measures). While it's quite subtle in the literature, at least for a mere structural and field geologist like myself, I believe the distinction is made by f(a) and a corresponding to the formalism for measures (with their generalized Reyni dimension) while D(h) vs h corresponds to the formalism for functions, with the holder exponent acting as a generalized Hurst exponent. The Frisch and Parisi method, referred to as the method of "incriments," had some pitfalls which quickly inspired A.Arneodo and JF Muzy to explore replacing incriments with wavelets in the late 80s, which could be thought of as a generalized oscillating box. A few years later, they addressed the shortcomings of the so-called "structure function" method, namely its inability to construct the right hand side of the singularity spectrum. Arneodo et al. 1995 developed a "canonical" approach to this problem through the thermodynamic formalism. The benefit of this method is that one can cover the slcale space in a Hausdorff like manner rather than a naive box counting per scale, and this makes the partition functions more robust to long range perturbations (if sufficient vanishing moments are used). However, there were still some issues to be addressed. Namely, the presence of negative holder exponents (for which the sup in the partition function would always be the finest scale) and also the presence of oscillating singularities. The negative holder exponents could be taken care of easily for 1D scalar signals / measures - just fractionally integrate by taking the cumulative sum and shift the spectrum to the left by 1. For signed signals and 2D + higher dimensional datasets, this was more problematic because it requires fractional integration. In theory, this should similarly shift the holder exponents. To avoid the edge effect generated via fractional integration, we can do a "pseudo fractional integration" by lifting the wavelet coefficients at each scale. 

The reason why fractional integration is not always a good magic trick is becasue of the presence of so-called oscillating singularities (chirp like singularities). One approach to handling this was introduced by Arneodo et al. in 1997 in their "Grand canonical" formalism, where the spectrum is characterized not only by holder exponent but also another exponent that characterizes the oscillating singularities. Strong oscillating singularities may appear naturally obvious in some datasets such as aeromatnetic flight lines. However, they are often subtle, even appearing in Zeta functions. We can also think of the relative orientations of the sensor and a structure (or a tilted outcrop in map view) -- periodic structures (not necessarily c_inf) observed at an angle appear chirp-like. The simplest example of this is taking a photo of a building with regular window spacing at an angle that is off 90° to teh face of the building. The windows further from the observer look like they are gettign closer together. This is an important consideration when thinking about analysis of crystal preferred orientation (CPO) in a 2D thin-section. 

The difficulties that arise with the fractional integration + oscillating singularities prompted Jaffard (1994, 1997 results about mf formalism for functions pt 1 and pt 2), and Meyer (1998 - wavelets, vibrations, and scaling) [also J+M 1996] to rigorously establish the multifractal formalism for functions. This was mostly inspired by a subject called microlocal analysis - inspired by JM Bony's "Second Microlocalization and Propagation of Singularities for Semi-Linear Hyperbolic Equations." 

As it turns out, Arneodo's method was not actually measuring the holder exponent but rather something dubbed by Meyer and Jaffard refer to as the "weak scaling exponent." The works of Jaffard and Meyer ultimately led to the SOTA for this type of analysis to be the so-called wavelet leaders method (leaders, p-leaders, and wse-leaders). The benefit of this DWT-based approach is that if the decomposition is dyadic, one can use dyadic boxes. However, in geological and geophysical signals, the singular properties of flucctuations of interest may be quite band-limited (e.g. smooth perturbations make it s.t. singular features must b e evauated at finest scales). Jaffard (1997 pt 1) actually outlines a "fix" for the WTMM method which turned into the leaders method; however, the CWT approach is not really used in practice. Probably becasue the DWT approach makes things simpler. In the presence of band-limited data (e.g. raster data, hyperspectral drill cores), a dyadic approach is not always appropriate and we may want to have bands with finer spacing than an octave (voices per octave). In this case, Jaffard's approach to CWT leaders involves taking a radial disc neighborhood, rather than boxes, at the WTMMM. Perhaps this can also be done for every pixel, but its computationally more of a nightmare. Nonetheless, I have attempted to put CWT leaders in this program with both options available, and it would be nice if someone with some expertise in microlocal analysis can validate or invalidate it.  


- **Multifractal analysis (canonical).**
Arneodo et al.'s 2-D wavelet transform modulus maxima method (WTMM):
  GeodynamX started as a GUI to wrap Kestener and Arneodo et al. and Bacry et al. software packages called xsmurf and lastwave. Eventually, I was able to benchmark my own python implementations against theirs in c, so GeoDynamX now relies little on the vendored packages. However, since we are still in our infancy, I use the GPL liscense to be able to use their algos if needed.
  
   Maxima chains, partition functions and the singularity spectrum D(h) can be easily calculated in this application. Concepts like the wavelet scalebar are introduced to help those with little background in this matter get familiar with the concepts and avoid reading mathematics which can be quite intimidating at first.
   
    Consider the use of any analying wavelet other than g1 and g3 experimental for now and please check the code for mistakes in the alternative wavelets here (in development). By alternative wavelets, I am referring to the tuneable wavelets such as teh fractional bspline / gaussian wavelets and the Tsallis q-deformed wavelets of Borges et al. (2004). Rather than making some kind of statement about the physics or long range correlations, I attempted to make these wavelets available in 2D (so there were some changes in the scaling function) so that we can tune the wavelets when it's practical. For example, the q-mexican hat as a smoothing function can be tuned to 2 where the negative phase segments of the Ricker wavelet are quite flattened. The analyzing wavelet is then a q-g3 and we are able to have an analyzing wavelet with 3 vanishing moments (to see through smooth perturbations/polynomial trends) that emphasizes true edges much more than the convolution of the negative phase of the analyzing wavelet about the edge (consider a binary line as an impulse in 2D space convolved with a g0 vs g1 vs g2 etc... and what the horizontal extrema would look like). 


    Also, DWT leaders 

    


- **Singularity analysis (microcanonical).** Per-pixel Hölder exponents from the gradient measure
  or from wavelet projections, the singularity spectrum by the histogram method, and
  reconstruction of the field from a band of exponents. This is the method of Turiel et al., and if one chooses to go the DWT route, the default display is a map of holder exponent from the generalized multifractal wavelet leaders projection (Cardo and Corvalan 2008). Both the multiaffine functional and measure (gradient first then smooth) routes are available, and the punctual regression or full scale regression are also options. When tuning the wavelet transform, a nested wavelet scalebar appears for ease of use. 
- **Multiscale edges.** The essence of geological mapping is that it is a form of information compression. As Shannon figured out, compression and prediction are mathematically equivalent. We like to use geological maps to make predictions, e.g. where an earthquake will occur or mineral deposit will be discoveered. Presumably, any sort of machine learning will be either biased or totally black-box unless it takes advantage of some connection to the underlying physics and chemistry. The WTMM method is especially intriguing because of the demonstration by Mallat and Zhong (1992) that multiscale edges contain most of the information from the original signal. This demonstration uses their dyadic spline wavelet which was probably used at the time for computational reasons (compact support enabled FIR filter to be  used). However, their method of projections + FFT can be used to recover a signal from WTMM edges from analysis inw hich the wavelet has no compact support. This goes back to Canny and Marr edge detectors. Meyer did show that they do not contain the complete information and demonstrated a wavelet that did, but it's kinda besides the point. 

There are also other edge detector tools like Perona–Malik anirostropic diffusion (perhaps the best performign), complex (discrete) cross-diffusion filtering (Araujo, Barbeiro et al), and a medial-axis extractor (Tang-You 2003, You et al. 2006). 



- **Decompositions.** Principal components of multi-band data, Tucker
  decomposition (generalized SVD) with delay-embedding capabilities, and 2-D singular spectrum analysis (Golyandina and Usevich)
- **Multiband data.** Sensor products such as ASTER import grouped by sensor and resolution.
  Each band can be analysed or removed on its own, and a stack displays as an RGB composite
  with per-band channel, solo and mute controls and ENVI-style stretches. The program should be able to read netcdf and hdf5, but if your dataset is not recognized, either message me or ask an AI agent to help (its 2026). 
- **Band routing.** A bus assembles bands from several datasets on the same grid into one
  stack for the multi-band tools, and a band stack can follow the results of other layers
  as they change. This might not be the best way to do things. I'm just a musician and wanted a program that works kinda like a DAW. 
- **Anisotropy.** Gradient-direction arrows on the wavelet maxima, and the angle statistics
  of Arneodo, Decoster and Roux (2000), in image coordinates or as map bearings.
- **Filters on chained structures like WTMM extrema.** By scale, orientation, modulus, chain length and topology.
- **Regions of interest.** Any tool can run on a rectangular region. The region is read with a
  margin, so tools that need neighbouring data see real data beyond its edges.

Tools form a chain: transforms first, then filters. Changing a filter only redraws; changing a
transform recomputes, and every result is cached. Advanced settings appear in the right column. Over there, one can modify something like the hyperparameters for chaining WTMMM extrema across scales and re-run the chaining portion without having to recompute the whole CWT, for example. 
**Distributed approximating functionals**
Were introduced for experimental purposes and for expanding and periodizing the domain for the disorder parameters to be implemented, but something aint right. Most geologic datasets are too messy to get a good boundary periodization. 
**Disorder parameter and generalized disorder parameter.**
In development (incomplete and not validated). 

**Wavelet ridges**
While we are quite interested in singular structures, represented by the wavelet "skeleton," we may also be interested in smooth structures. This tool needs some benchmarking, but the goal is wavelet ridge extraction using the algo from the appendix of Passot and Newell (1994) in their paper on universal patterns. 
**Dyadic interpolation**
Deslauriers and Dubuc algo. The tool is complete to the best of my knowledge, but I do need to make some changes to the user interface to enable all 4 hyperparameters. This is a great tool when used correctly. 

## Data

GeoTIFF (also inside a `.zip`), netCDF, HDF5 and HDF4 grids, and common image formats such as
PNG and JPEG are supported. I plan to have more of my colleagues test the software and give me feedback on types of file formats to include (like SEGY, LAS, etc). 

The goal of this software was once speficially tailored to microstructural analysis, especially EBSD datasets. The next step is to use BlueQuartzSoftware/EbsdLib to support these file formats and operations in SO(3). I have used orix and also MTEX, and teh results are not always teh same between the two. MTEX functions can be called in python using the matlab engine. However, since BlueQuartz has been benchmarked against MTEX, it's safe to go ahead and not assume that everyone has or wants to have a matlab license. Since I am using AI agents to make the GUI, I only have so much domain knowledge to ensure that certain aspects of the user experience like caching signal routing are working as efficiently as possible. To avoid making a bloated program, I have decided that perhaps moving forward I can make new repos for GDX packages. E.G. and EBSD package can build off of EBSDLib to efficiently enable operating on those datasets. A seismology package can import the relevant tools. Etc etc... so we have the GDX core app and domain-specific extension packages. I have already forked Oliver Philcox's encore package for fast NPCF and created a branch enabling GPU speedup. 

## Install and run

GeoDynamix needs Python 3.12 and runs on Apple Silicon macOS and on Windows. Install it into an
environment of its own:

```bash
git clone https://github.com/MarioKartUltra/GeoDynamX.git
cd GeoDynamX
conda create -n geodynamix python=3.12
conda activate geodynamix
pip install -e ".[app]"
python -m dynamix.shell.app
```

## Compute settings

Every FFT runs on mlx (the GPU) on Apple Silicon, and on FFTW3 everywhere else. In Settings you
can choose the engine and the precision, 32 or 64 bit. 64-bit FFTs always run on FFTW3, because
mlx has no double-precision FFT. Changes take effect after a restart. The failing tests are pyvista tests and for now the 3D view is not really necessary, although I plan to fix that in the future. For that reason, I am pretty sure this app will work on mac, linux, and windows since lin and win are passing all the same tests that mac passes. If someone has a machine with CUDA and would like to contribute by making or adding support for a CUDA implementation of the FFT, be my guest. I use a macbook, so I cannot ensure this works on Windows at all (even with FFTW3) and would love some windows users to test it and give feedback. Since the Apple silicon chip does not support 64 bit operations on the GPU, it defaults to 32 bit and if you really need 64 bit, you can use FFTW3. I cannot think of a single sensor used in geosciences whose AD converter is 32 bits and higher. Even seismometers record 24 bit and store 32 bit digitally, so the "threshold" selection should be able to precisely remove irrelevant information at 32 bits if one uses a white nosie dither at +/- 1 LSB. 

The only tool where it might be worth skipping the metal framework and going pure CPU on a mac is the n-point correlation functions, especially the 3-6 point correlation functions. Those are not yet available in the app, so sit tight. 

## Documentation

The [device reference](docs/reference/devices_reference.pdf) describes every tool and parameter,
multiband data and band routing, the wavelet normalisation, and the wavelet families, with
figures drawn from the application's own kernel code.

## Status

GeoDynamix is in beta: tools, parameters and the saved-project format may still change.

## License and citation

Copyright (C) 2026 Abraham Joseph Okayli Masaryk. GeoDynamix is free software under the GNU
General Public License; see [`LICENSE`](LICENSE). The author's code is GPL-2.0-or-later. The
modules translated from xsmurf (Decoster, Kestener, Roux and Arneodo; GPL-2.0) are GPL-2.0-only,
and the 1-D WTMM code is based on LastWave (Bacry and Mallat; GPL-2.0-or-later).
[`NOTICE`](NOTICE) lists the derived code and the principal scientific references, and
[`CITATION.cff`](CITATION.cff) explains how to cite this software.
