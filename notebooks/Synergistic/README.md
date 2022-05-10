# Synergistic notebooks

You should find in this directory a few notebooks that are designed to give you a feeling for synergistic reconstructions.

0. [Gradient descent/ascent for MR, PET and CT](#gradient-descent-for-MR-PET-CT)
1. [Generate the data](#gen_data)
2. [de Pierro MAPEM with the Bowsher prior](#de_pierro)
3. [Dual PET](#dual_pet)
4. [HKEM](#HKEM)
5. [Joint-TV for MR](#Joint-TV)
6. [Joint-TV for PET](#Joint-TV_P)
7. [Joint-TV for PET/SPECT](#Joint-TV_PS)

## 0. Gradient descent/ascent for MR, PET and CT<a name="gradient-descent-for-MR-PET-CT"></a>

The [gradient_descent_mr_pet_ct notebook](gradient_descent_mr_pet_ct.ipynb) shows how to write
a simple gradient descent (or ascent...) algorithm for MR, PET and CT (using CIL for the latter).
It is not really a "synergistic" notebook in itself, but can serve as the basis for any synergistic algorithm that uses alternating optimisation between the different modalities.

## 1. Generate the data <a name="gen_data"></a>

#### Get the data

The file, [BrainWeb.ipynb](BrainWeb.ipynb), will generate data that will be used by a few of the notebooks. This uses a wrapper that obtains the brainweb data, installed as part of the requirements of the exercises.

#### What images?

The extracted images include two PET images (FDG and amyloid), two MR images (T1 and T2) and the corresponding mu-map.

#### Convert to STIR

The script will convert the native brainweb data into STIR interfile format. It will also create cropped versions of the files, which have the suffixes `_small`. Most of the notebooks use these to speed up reconstruction times. But you could equally use the full-size images.

#### Forward project
The images are then forward projected, and the resulting noise and noiseless sinograms are created.

#### Misalignment

A misalignment is also added to the amyloid image (and its corresponding mu-map), and the noise and noiseless sinograms are created and saved. The effect of misalignment is discussed in the [Dual_PET.ipynb](Dual_PET.ipynb) notebook.

#### Adding tumours to the data

Lastly, a tumour is added to the original amyloid image. This data is useful for studying the effect of feature suppression when the feature is not present in the side information.

## 2. de Pierro MAPEM for Bowsher <a name="de_pierro"></a>

[MAPEM_Bowsher.ipynb](MAPEM_Bowsher.ipynb) is a continuation of [../PET/MAPEM.ipynb](../PET/MAPEM.ipynb), however this example uses a (quadratic) Bowsher prior that depends on side information (in this case an MR image) as opposed to a quadratic prior using uniformm weights.

## 3. Dual PET <a name="dual_pet"></a>

In [Dual_PET.ipynb](Dual_PET.ipynb), we reconstruct the FDG and amyloid acquisitions at the same time. This is an extension of [MAPEM_Bowsher.ipynb](MAPEM_Bowsher.ipynb), in which the Bowsher weights are constantly updated as our estimate of the other acquisition improves. 

The further complication of having a misalignment between the FDG and amyloid is also included. 

Answers for this notebook is given in [Solutions/Dual\_PET\_Answer-noMotion.ipynb](Solutions/Dual_PET_Answer-noMotion.ipynb) and [Solutions/Dual\_PET\_Answer-wMotion.ipynb](Solutions/Dual_PET_Answer-wMotion.ipynb).

## 4. HKEM <a name="HKEM"></a>

The hybrid kernel EM method is explored in [HKEM_reconstruction.ipynb](HKEM_reconstruction.ipynb). The HKEM algorithm is an example of a "guided" reconstruction method. Here, an MR image is used to guide the PET reconstruction. The effect on many parameters, such as neighbourhood size is also explored.

## 5. Joint-TV for MR <a name="Joint-TV"></a>

A joint total variation regulariser is explored in [cil_joint_tv_mr.ipynb](cil_joint_tv_mr.ipynb) for the purpose of multi-modal MR reconstruction.

## 6. Joint-TV for PET <a name="Joint-TV_P"></a>

A joint total variation regulariser is explored in [cil_joint_tv_PET.ipynb](cil_joint_tv_PET.ipynb) for the purpose of multi-modal PET reconstruction.

## 7. Joint-TV for PET/SPECT <a name="Joint-TV_PS"></a>

A joint total variation regulariser is explored in [cil_joint_tv_PET_SPECT.ipynb](cil_joint_tv_PET_SPECT.ipynb) for the purpose of multi-modal reconstruction.
