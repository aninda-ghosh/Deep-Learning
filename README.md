# Deep-Learning-OpenCV

---

## Table of contents

| Name                                  | Description                                                                                               |
| --------------------------------------|---------------------------------------------------------------------------------------------------------- |
| Classification                        | The code uses the Breast cancer Dataset from SKlearn for training the model                               |
| Regression                            | We find the curve which fits the Moore's Law                                                              |
| 3-input, 1-output node Neural Network | We will implement a NN which will incorporate 3 inputs and 1 output with tanh as the activation function  | 
| MJPEG Stream Video Capture            | This implementation is under Smart Home-Antitheft category.                                               |

--- 

## **Guidelines**

- Following semantic versioning with ```Major.Minor.Patch```
- Every small changes will come under ```patch``` version. (bugfix, bug)
- Every feature introduction will change the ```minor``` version.
- Every interface logic change will change the ```major``` version.
- Example of the versioning
  - 2.0.005
    - Major version - 2
    - Minor Version - 0
    - Patch Version - 005
  - Always create a new branch for a feature introduction and raise a pull request accordingly.
  - Releases will be tagged either ```Pre Release``` or ```Latest Release``` from the main branch only.
  - Git commits will have the changelogs description as the commit message. e.g. **2.0.005 (feature+update)**
  - Definition of keywords
    - **feature** - Any introduction of a new feature
    - **bug** - With the commited change the system is in a buggy state.
    - **bugfix** - for any patches applied over a bug.
    - **update** - general updates like updating readme. (this won't increment any version numbers)
    - **experimental** - This stays out of the main branch unless the experiment is solidified to create a feature out of it.

---

## Changelogs

### MJPEG Stream Video Capture

#### 0.0.1 (feature)

- Getting stream from the MJPEG URL being broadcasted from the ESP32 CAM.
- Applying Contrast Limited Adaptive Histogram Equalization method for adaptive equalization.
- Plotting the generated histograms based on the method.







