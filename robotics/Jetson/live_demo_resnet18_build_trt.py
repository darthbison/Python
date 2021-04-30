#!/usr/bin/env python
# coding: utf-8

# # Collision Avoidance - Build TensorRT model for live demo
# 
# In this notebook we'll use the model we trained to detect whether the robot is ``free`` or ``blocked`` to enable a collision avoidance behavior on the robot.  
# 
# ## Load the trained model
# 
# We'll assumed that you've already downloaded the ``best_model.pth`` to your workstation as instructed in the training notebook.  Now, you should upload this model into this notebook's
# directory by using the Jupyter Lab upload tool.  Once that's finished there should be a file named ``best_model.pth`` in this notebook's directory.  
# 
# > Please make sure the file has uploaded fully before calling the next cell
# 
# Execute the code below to initialize the PyTorch model.  This should look very familiar from the training notebook.

# In[ ]:


import torch
import torchvision

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda().eval().half()


# Next, load the trained weights from the ``best_model_resnet18.pth`` file that you uploaded

# In[ ]:


model.load_state_dict(torch.load('best_model_resnet18.pth'))


# Currently, the model weights are located on the CPU memory execute the code below to transfer to the GPU device.

# In[ ]:


device = torch.device('cuda')


# # TensorRT

# > If your setup does not have `torch2trt` installed, you need to first install `torch2trt` by executing the following in the console.
# ```bash
# cd $HOME
# git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# cd torch2trt
# sudo python3 setup.py install
# ```
# 
# Convert and optimize the model using torch2trt for faster inference with TensorRT. Please see the torch2trt readme for more details.
# 
# > This optimization process can take a couple minutes to complete.

# In[ ]:


from torch2trt import torch2trt

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)


# Save the optimized model using the cell below

# In[ ]:


torch.save(model_trt.state_dict(), 'best_model_trt.pth')


# ## Next
# 
# Open live_demo_resnet18_build_trt.ipynb to move JetBot with the TensorRT optimized model.

# In[ ]:




