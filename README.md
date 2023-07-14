# A Deep Registration Method for Accurate Quantification of Joint Space Narrowing Progression in Rheumatoid Arthritis

Rheumatoid arthritis (RA) is a chronic autoimmune inflammatory disease that leads to progressive articular destruction and severe disability.Joint space narrowing (JSN) progression has been regarded as an important indicator for RA progression and has received significant attention.Radiology plays a crucial role in the diagnosis and monitoring of RA through the assessment of joint space.
A new framework for monitoring joint space by quantifying joint space narrowing (JSN) progression through image registration in radiographic images has emerged as a promising research direction. This framework offers the advantage of high accuracy; however, challenges still exist in reducing mismatches and improving reliability.In this work, we utilize a deep intra-subject rigid registration network to automatically quantify JSN progression in the early stage of RA. In our experiments, the mean-square error of the Euclidean distance between the moving and fixed images is 0.0031, the standard deviation is 0.0661 mm, and the mismatching rate is 0.48%. Our method achieves sub-pixel level accuracy, surpassing manual measurements significantly. The proposed method is robust to noise, rotation, and scaling of joints. Moreover, it provides loss visualization, which can assist radiologists and rheumatologists in assessing the reliability of quantification, and has scope for future clinical applications.As a result, we are optimistic that this proposed work will make a significant contribution to the automatic quantification of JSN progression in RA.


## File Description
JSNmeasurement: Package of segmentation and registration tasks. It is being further refined and optimized.<br/>
Segmentation: For segmentation task.<br/>
Sample: We provide a set of images to help better understand the registration algorithm of our approach.<br/>
Others: For registration task.<br/>

## How to use registration method
Please run the predict.py file. And modify the image input paths, including the fixed and moving paths and their corresponding masks paths. The output will be the display mode we have set.
