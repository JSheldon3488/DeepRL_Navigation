[//]: # (Image References)

# Project 1: Navigation

### Environment Explanation

This project trains an agent to navigate (and collect bananas!) in a large square world. A video of the environment is below.

<p align="center">
    <img src = "https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif">
</p>

A **reward** of +1 is provided for collecting a yellow banana, and a **reward** of -1 is provided for collecting a blue banana.  The goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The environment is set to run for 300 timesteps (actions) as a default.

The **state space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  The **action space** consists of four discrete actions:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

**Goal:** The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Setup
1. Download all the dependencies
    * [OpenAI gym.](https://github.com/openai/gym) Install instructions in the repository README.
    * [Udacity Deep RL Repo.](https://github.com/udacity/deep-reinforcement-learning#dependencies) Install instructions in dependencies section of README. Details for just setting up this repo below.
        ```bash
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install .
        ```

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in the `p1_navigation/` folder in the Udacity Deep RL repository (from step one), and unzip (or decompress) the file.

### Training Agents

Follow along in the [Navigation Notebook](Navigation.ipynb) to get started with training agents! Details about all the currently implemented agents can be found in the report [here](report.md).