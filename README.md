# Atmospheric retrievals of K2-18b JWST observations for Wogan and Batalha (2025)

This archive contains the code needed to reproduce the Wogan and Batalha (2025) atmospheric retrievals of James Webb Space Telescope (JWST) observations of the K2-18b exoplanet. The code first reproduces the dimethyl sulfide (DMS) and/or dimethyl disulfide (DMDS) atmospheric detection argued for in [Madhusudhan et al. (2025)](https://doi.org/10.3847/2041-8213/adc1c8), when considering MIRI LRS observations in isolation. The code then does additional retrievals to show that DMS and/or DMDS is not detected when considering all available JWST observations.

All calculations and Figure 1 in Wogan and Batalha (2025) are reproduced in two steps as outlined below:

## Step 1: Installation and setup

If you do not have Anaconda on your system, install it at this link or in any way you prefer: https://www.anaconda.com/download . Next, run the following code to create a conda environment `k218b_dms`.

```bash
conda create -n k218b_dms -c conda-forge -c bokeh python photochem=0.6.5 numpy=1.24 matplotlib pandas pip numba xarray pathos threadpoolctl bokeh=2.4.3 mpi4py wget unzip tar ultranest
conda activate k218b_dms
```

Next, run the code below to install and setup PICASO v3.1.2.

```bash
pip install picaso==3.1.2

# Opacity DB
wget https://zenodo.org/records/15299348/files/lupu_0.5_15_R10000.db.tar.gz
tar -xvzf lupu_0.5_15_R10000.db.tar.gz
rm lupu_0.5_15_R10000.db.tar.gz

# Get reference
wget https://github.com/natashabatalha/picaso/archive/4d90735.zip
unzip 4d90735.zip
cp -r picaso-4d907355da9e1dcca36cd053a93ef6112ce08807/reference picasofiles/
export picaso_refdata=$(pwd)"/picasofiles/reference/"
echo $picaso_refdata
rm -rf picaso-4d907355da9e1dcca36cd053a93ef6112ce08807
rm 4d90735.zip

# Get the star stuff
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"
echo $PYSYN_CDBS
rm synphot3.tar.gz

# Get more star stuff
wget https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
tar -xvzf hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
mv grp/redcat/trds/grid/phoenix picasofiles/grp/redcat/trds/grid/
rm -r grp
rm hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar

```

## Step 2: Run the code and plot

The code below sets up the environment then runs the retrieval script with MPI. This archive includes the completed retrievals in the `ultranest/` directory, so this script will take very little time because the code will resume from completed calculations.

If you run the code from scratch (i.e., if you delete the contents of the `ultranest/` directory), then this calculation will take a lot of time and require a super computer. I used 600 cores (25 nodes each with 24 cores) on the NASA Pleiades supercomputer, and the calculation took about ~20 hours. The example command below uses 4 MPI processes. You can set the number of MPI processes by changing `-n 4` to `-n <number of processes>`.

```bash
# environment setup
conda activate k218b_dms
export picaso_refdata=$(pwd)"/picasofiles/reference/"
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"

# run the main script
mpiexec -n 4 python retrieval_run.py > retrieval_run_output.txt
```
Finally, you can plot some results with the following. See the figures in `figures/`.

```bash
python plot.py
```