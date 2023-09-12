#!/bin/sh
#SBATCH --time=8-00
#SBATCH --begin=now
#SBATCH --gres=gpu:1
#SBATCH --job-name=convert_annotations
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=gpu


module load cuda/11.1
source activate vatrack



#Conversion to SAILVOSCut
srun python sailvos2coco_cut.py -i ./data/sailvos/annotation_24_classes_before_conversion -o ./data/sailvos/annotation_24_classes_conversion_after_conversion
