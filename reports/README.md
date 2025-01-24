# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [x] Setup cloud monitoring of your instrumented application (M28)
* [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [x] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

MLOPS 41

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s233353, s195849, s180351

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We utilized the Hugging Face Diffusers library, a powerful third-party package, to enable generation of high-quality pixel art images in our project. Specifically, we leveraged key components like Denoising Diffusion Probabilistic Models (DDPMscheduler), UNet2DModel, and optimization.get_scheduler to perform crucial tasks - model denoising, training, and learning rate optimization. These advanced machine learning capabilities from the Diffusers library were pivotal in enhancing our pixel art generation, allowing us to create sophisticated, visually striking results that met the project's requirements. 


## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We updated our requirement.txt in parallel with adding libraries to our code together with pyproject.toml, it was our way to manage the dependencies in our project. We also have a pyproject.toml which will go through the necessary steps to copy our environment. 

Step 1. Clone our repository.

Step 2. Create your own blank environment. 

Step 3. run 'pip install -e .' to copy our environment dependencies to your environment. 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We used cookiecutter template and filled out all of the folders.
We also added a dvc folder for data version control. In the root folder, we have added cloudbuild.yaml for the GCP and a ruff_cache folder for code check for pull request to GitHub. 


### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We implementedcode quality and formatting rules using 

- Ruff for linting and formatting
- mypy for static type checking
- Pre-commit hooks to enforce these standards

This can be seen in our GitHub workflow and it automatically run these checks on every pull request, ensuring consistent code quality.

These practices are crucial in larger projects because they prevent technical debt, improve code readability, and catch potential errors early. Type checking helps prevent runtime errors, while consistent formatting makes collaboration smoother. Documentation and typing act as built-in code documentation, making the codebase more maintainable and easier for new team members to understand.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 4 test files; data, model, train and api. 

- For data, we tested the if the path which contained the data existed, then tested that the images in the 3rd label has the correct amount together with the correct data format. 

- For model, we test if a model checkpoint exists, then tested that the model was correctly initialized and generated sample from the model has the correct output shape. 

- For training, we test if the training directory correctly points to the model directory, then tested that the training ran succesfully with dataset subset with 1 epoch. 

- For api, we test if the app succesfully starts, a small load test and can generate an image from the model that we worked with.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our total code coverage is at exaclty 50%, which is not nearly enough for us to trust that it is error or bug free. Our highest code coverage percentage for the individual file outside of init, is cloud_function.py at 68% which is responsible for loading the model from GCP and communicationg with the cloud to get samples and visualize them in addition to storing them. The lowest code coverage percentage is 23% for generate_sample.py which is responsible for taking a model and generate samples. From the coverage report, it is clear that additional tests are needed to get 100% code coverage, but even then, it is still difficult to fully trust that the code will be error free as the complexity of the tests depends on our ability to test for edge-cases. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Our workflow utilized branches and pull requests for version control. The '.yaml' files show multiple workflows triggered on both push to the main branch and 'pull_request' to the main branch. This approach allows team members to work on separate branches, making independent changes without directly modifying the main branch.

Pull requests serve as a critical code review mechanism. Before merging any changes into the main branch, our workflows run comprehensive checks including unit tests, type checking (mypy), code linting (Ruff), and data synchronization tests across different operating systems and Python versions. 
This ensures code quality, catches potential issues early, and maintains the integrity of the main branch by requiring all checks to pass before merging.

The pre-commit CI workflow further enhances our version control by automatically checking and potentially fixing code formatting and style before commits are finalized.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We use DVC to manage our data in the project. The data that we have contains 89000 images, thus zipping it to archive.zip was necessary to utilize DVC efficiently. The version control is improved as using the GitHub DVC workflow automatically syncs changes and updates the data directory. This streamliness our project's data management. 


### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continuous integration into 5 separate workflow files, each handling a distinct aspect of our testing and deployment pipeline:
The first workflow handles unit testing (tests.yaml) across multiple operating systems (Ubuntu, Windows, macOS) and Python versions (3.10, 3.11). It utilizes pip caching for efficient dependency management and runs coverage reports through pytest.
The second workflow (codecheck.yaml) manages code quality through Ruff for linting and formatting, along with mypy for type checking. Like our testing workflow, it runs on multiple OS/Python combinations with caching enabled.
The third workflow (cml_data.yaml) focuses on our data pipeline, validating our data processing across platforms. It handles Google Cloud authentication and DVC data pulling to ensure our data processing remains consistent.
For documentation, we have a dedicated workflow (docs.yaml) that automatically deploys our MkDocs documentation when changes are pushed to main.
Finally, our pre-commit workflow (pre_commit.yaml) enforces code quality standards before commits and can automatically fix formatting issues.
All workflows utilize GitHub Actions' matrix testing capabilities for cross-platform compatibility and dependency caching to optimize execution time. An example of our testing workflow can be seen [here](https://github.com/Gyllz98/pixel_art_diffusion/actions/runs/12949125106/job/36119190985), which demonstrates our comprehensive approach to ensuring code quality and reliability.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used typer parsing as the CLI. It included run_name, epochs and label_subset. As an example:
train --epochs 100*


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made a config file where we specified the model parameters, training parameters, data paths and wandb logging parameters in a yaml file. This was then parsed by hydra.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

The screenshots show training metrics from my diffusion model experiments tracked in Weights & Biases. I'm tracking:

Batch-level metrics:

- Learning rate: Monitors optimizer behavior and learning schedule
- Loss: Shows model convergence
- Epochs: Tracks training progress

Epoch-level metrics:

- Average loss: Indicates overall training stability
- Epoch count: Shows training duration

These metrics are important because they help identify training issues early, verify the model is learning properly, and ensure hyperparameters like learning rate are appropriately tuned. The smooth downward trend in loss curves and stable learning rate decay suggest successful training. But as we have spend the vast majority of the time on cloud, api, and CI/CD we did not run many experiments. Aditionally, we did prototyping in a notebook locally until training was stable.

[](figures/wandb.jpeg)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <docker.io/darius0429/pixel-art-app>*
>
> Answer:

We dockerized the training script and the api script. One has to build the image with:
docker build -t pixel-art-app:latest -f dockerfiles/api.dockerfile .

And run it with:
docker run -p 8080:8080 pixel-art-app




### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

> *We added a lot of logging to our script for debugging to see which parts fail. These served as print statements.
We did profile our training script and generate_sample script to see what parts require optimization the most.*

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

- Bucket: We use Bucket to store our data, model, and generated images for drift detection.
- Functions: We made a cloud Function to load our model from the bucket. We also made one that does different tasks related to datadrifting such as pushing and pulling generated images.
- Artifact Registry: Holds our docker images.
- Build + trigger: Everytime we time new code is pushed to main we rebuild the the api and deploy it.
- Run: As stated above our api is deployed with cloud Run so that you only need to launch the frontend locally.
- Monitoring: Monitors cpu usage among other things of our application.
- Alert: as generation is linked to uploading to the bucket we have set up an alert that triggers if too many uploads are made to the bucket within half an hour.
- Compute Engine: Our model is trained in a VM using L4 gpu.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

The main way we used Google Cloud Platform's Compute Engine was for training our diffusion model. We provisioned a VM instance with an NVIDIA L4 GPU, which offers good performance for ML workloads while being cost-effective. The VM was configured with 16 vCPUs and 64GB RAM to handle our data preprocessing and training pipeline efficiently.
The instance was launched using our custom Docker container (defined in dockerfiles/train.dockerfile) which contained all necessary dependencies including PyTorch, diffusers library, and our training code. We used Debian as the base OS since it works well with ML workloads.
The configuration was automated through Cloud Build (as seen in cloudbuild.yaml), allowing us to consistently deploy our training environment. The L4 GPU proved sufficient for training our 16x16 pixel art model, offering a good balance between cost and performance for our relatively small image generation task.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[](figures/bucketshot1.jpeg)
[](figures/bucketshot2.jpeg)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[](figures/ARshot1.jpeg)
[](figures/ARshot2.jpeg)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[](figures/cloudbuild.jpeg)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We successfully trained our pixel art diffusion model in Google Cloud Platform using Compute Engine. Our training pipeline was implemented through a multi-stage process using Cloud Build and custom Docker containers. First, we prepared our training container using train.dockerfile which included PyTorch, the diffusers library, and our custom training code. The training data was stored in a Cloud Storage bucket for efficient access.
We chose Compute Engine over Vertex AI because we needed fine-grained control over the GPU configuration and training environment. Our custom training pipeline required specific dependencies and data preprocessing steps that were easier to implement in a custom container. Additionally, Compute Engine provided better cost optimization for our use case, as we could precisely control when the GPU instance was running and shut it down automatically when training completed.
The flexibility of Compute Engine allowed us to implement features like automated model checkpointing to Cloud Storage and seamless integration with our monitoring stack, which proved essential for maintaining our MLOps pipeline.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We implemented a comprehensive API using FastAPI for our pixel art generation model. The API (api.py) features several key endpoints: /generate for creating new pixel art images, /samples for retrieving stored samples, /model_info for model status, and /data_drift for monitoring data quality.

We enhanced the API with production-ready features:

- Prometheus metrics tracking generation requests, memory usage, and latency
- Background task handling for storing generated samples
- Proper error handling and logging
- Data drift monitoring using Evidently AI
- Async context manager for model loading
- Streaming responses for image delivery
- Health check endpoint

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We successfully deployed our API using Google Cloud Run for scalable, serverless deployment. The deployment process is automated through our cloudbuild.yaml pipeline, which:
1. Builds the Docker container from api.dockerfile
2. Pushes it to Google Container Registry 
3. Deploys to Cloud Run with automatic scaling

The API is accessible at https://pixel-art-api-146224029275.europe-west1.run.app. Users can interact with it through:

1. Web Interface (Streamlit):
```bash
pixel-art-app
```
2. Direct API calls:
```bash
# Generate 4 pixel art images
curl -X POST "https://pixel-art-api-146224029275.europe-west1.run.app/generate?num_samples=4" -o generated_art.png

# Get model info
curl https://pixel-art-api-146224029275.europe-west1.run.app/model_info
```
Cloud Run automatically handles scaling based on demand, and we only pay for actual usage. The service integrates with Cloud Monitoring and Logging for observability.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We conducted comprehensive load testing using Locust, simulating 100 concurrent users performing different API tasks. The test prioritized three main tasks:

1. Loading the page and model (highest priority)
2. Retrieving model information
3. Generating pixel art samples (most resource-intensive)

The load testing results showed interesting patterns in API performance under stress:

- Mean response time increased proportionally with user load
- The model generation endpoint showed the highest latency, as expected
- The API demonstrated good stability, never crashing despite the load
- Cloud Run's auto-scaling helped maintain service availability

The test results helped us identify potential bottlenecks and validate our deployment architecture. While response times increased under load, the service remained operational throughout the test, demonstrating the robustness of our Cloud Run deployment and the effectiveness of our async request handling.

For unit testing, we used pytest to test individual API endpoints and model functionality, ensuring reliable service operation.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We implemented Prometheus metrics in our API but didn't complete the cloud deployment of the monitoring stack due to time constraints. Our monitoring implementation includes:

- Request metrics: `pixel_art_generate_requests_total`
- Performance metrics: `pixel_art_generation_time_seconds`
- Resource usage: `pixel_art_memory_usage_bytes`
- Data drift monitoring using Evidently AI

While these metrics are accessible through the API's `/metrics` endpoint, we couldn't complete the deployment of a Prometheus sidecar container in Cloud Run to collect and store them long-term. With more time, we would have:

1. Deployed Prometheus alongside our API
2. Set up regular dashboard for visualization
3. Implemented alerts for anomalies

This would have enabled us to track model performance degradation, resource utilization patterns, and data drift over time, crucial for maintaining service quality and planning capacity.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:
We worked in one cloud for the whole project. We spent 5.44 USD in total. Compute engine training (renting the L4 GPU) drained the credits the most by far. Working in the cloud is very tedious and time consuming, however we understand the benefits of it when thinking about scaling applications.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented a frontend and a datadrift monitoring system to the build. 

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

[](figures/overview.jpeg)
- The starting point of the diagram is our local setup, where we integrated DVC for data management, Hydra for configuration, and the core model components (data.py, model.py, train.py) into our code.
- Whenever we commit code, pre-commit hooks run first to check linting and code formatting. After pushing to GitHub, it automatically triggers our CI/CD workflows which include Unit tests (data, model, and API tests), Code quality checks (PEP8 and type checking), Coverage reporting and Dependabot checks for dependencies.
- From there, if all checks pass, the diagram shows a build trigger that initiates the Docker image build and pushes it to the Container Registry in Google Cloud Platform.

>For the training flow:

- The training pipeline pulls data using DVC from Cloud Storage Runs the training with configurations from Hydra Logs metrics and model artifacts to Weights & Biases Model artifacts are stored back in Cloud Storage.

>For the serving flow:

- Cloud Run pulls the latest Docker image from Container Registry
It runs the FastAPI backend service
The Streamlit frontend interacts with this backend
When users generate images, they're stored back in Cloud Storage
The entire service is monitored through Cloud Monitoring with alerts set up

- All components are connected through GCP, with Cloud Storage acting as a central hub for data, models, and generated images, while Cloud Run handles the serving infrastructure and Cloud Monitoring keeps track of system health.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

> This project the toughes challenge we faced was deploying it to the cloud. We had to spend an exorbitant amount of time in cloud pulling, pushing and deploying the trigger when we pushed to main. This was due not being able to access stuff beacause of policies. Also setting up code-checks and other workflows slowed down the development process quite a bit. Also if these failed it meant a 10 minute setback each time. 

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s233353 was in charge of building the docker images both for training and the api, integrating configuration experiments, profiling, setting up CI workflows for the git repo (pre-commit, code-check, dvc code sync, unittests), building the front end, data drift analysis and some parts of deploying it to cloud.

Student s195849 was in charge of everything GCP: Compute Engine training, Continous Deployment with Cloud Run/Build, Buckets (and in extension dvc), etc. Also finalized CI (pre-commit, code-Check, unittests). Also set-up the repo and wrote the model, training loop, and dataloader + set-up wandb logging. Also, did the documentation.
