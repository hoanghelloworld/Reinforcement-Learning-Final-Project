# MAgent2 RL Final Project

## Overview
This project trains a multi-agent RL model in the MAgent2 `battle` environment. The code sets up agents and evaluates them against:  
- Random Agents  
- A Pretrained Agent  
- A Final (stronger) Agent  

For each setting, the performance is measured by rewards and win rates.

The figure below depicts the results when the VDN mixer model competes with random opponents, pretrained agent, final pretrained agent.

![Caption: vdn_mixer VS Random Agent](https://github.com/user-attachments/assets/796d2134-90ea-4cac-b9bf-d51e1ff0aa9f)


![vdn_mixer VS redpt Agent](https://github.com/user-attachments/assets/14a048ce-ec38-4809-8d85-77bcd1d7772f)


![vdn_mixer VS final Agent](https://github.com/user-attachments/assets/ac1f0f21-b968-45cd-bd09-509223fc9c88)


## Installation
pip install -r requirements.txt
## Source Code Summary
- **`dqn_train/dqn.ipynb`**: Main training script for the DQN-based agent.
  
  [![Kaggle Notebook](https://github.com/user-attachments/assets/fa91a220-a957-49aa-b023-66f4dbcd19d6)](https://www.kaggle.com/code/huyhonglo/dqn-ipynb)
  
  you need to "copy&edit" on kaggle to run script, then upload the red.pt file to kaggle and correct the path in the source code
  
- **`dqn_train/dqn_noise_network.ipynb`**: Main training script for the DQN with Noise Network agent.
  
    [![Kaggle Notebook](https://github.com/user-attachments/assets/fa91a220-a957-49aa-b023-66f4dbcd19d6)](https://www.kaggle.com/code/huyhonglo/dqn-noise)

    you need to "copy&edit" on kaggle to run script, then upload the red.pt file to kaggle and correct the path in the source code
  
- **`vdn_mixer_train/vdn_mixer.ipynb`**: Main training script for the VDN agent.
  
    [![Kaggle Notebook](https://github.com/user-attachments/assets/fa91a220-a957-49aa-b023-66f4dbcd19d6)](https://www.kaggle.com/code/huyhonglo/vdn-mixer)

    you need to "copy&edit" on kaggle to run script, then upload the red.pt file to kaggle and correct the path in the source code
  
- **`test_model/eval_test/test_dqn(others same).ipynb`**: Evaluation code against different opponents.  
you can find checkpoints at source model/dqn/checkpoints or model/vdn_mixer/
## Report
Survey_on_MAgent2_Battle_Using_DQN_Variants_Report_Group22.pdf
- Experimental setup  
- DQN variant details  
- Performance comparison and discussion  

## Demos
For additional demos, see the `video` folder 

## References
1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/MAgent2)  
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)  

For environment setup and agent interaction details, refer to the MAgent2 documentation.  
