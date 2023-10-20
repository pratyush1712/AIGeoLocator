# Graft - Flask based web application for Demonstration

This guide provides detailed instructions on how to set up, run, and deploy the Flask based web application for demonstrating the Graft project.

## Initial Setup

1. **Prepare the Model Folder**:
   - Place the `data.txt` file in the `model` folder.
   - Place the `MA_2020.npz` file in the `model` folder.

2. **Environment Configuration**:
   - Place the `.env` file in the root directory of the project. Ensure all necessary environment variables are set in the `.env` file for proper configuration.
    ```
        export FLASK_ENV=production
        export IMAGE_SOURCE=https://research.cs.cornell.edu/caco/data/graft/MA/
        export DATA_SOURCE=https://cv.cs.columbia.edu/utkarsh/mapper/MA_2020.npz
    ```

3. **Virtual Environment (venv) Setup**:
   - It's recommended to use a virtual environment to isolate the project dependencies. To set up a virtual environment, navigate to the project root and run:
     ```bash
     python3 -m venv venv
     ```
   - Activate the virtual environment:
     - On macOS and Linux:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       .\venv\Scripts\activate
       ```
   - Install the required dependencies using the provided `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

## Running the Application Locally

- To run the application, navigate to the project root and run:
    ```bash
    python app.py
    ```
- The application will be accessible at `http://localhost:8080/`.

## Deploying the Application

1. ### Gunicorn
   - To deploy the application, navigate to the project root and run:
       ```bash
       gunicorn --bind 0.0.0.0 app:app
       ```
   - The application will be accessible at `http://<server-ip>:8000/`.

2. ### Docker
    If you're using Docker for deployment, follow the steps below:

    1. For Local Testing:
        
        - Build the Docker image for local testing using the following command:
            ```bash
            docker build --build-arg FLASK_ENV=development -t graft .
            ```
            This command sets the Flask environment to development mode and builds the Docker image with the tag graft.

        - Run the Docker container using the following command:
            ```bash
            docker run -p 8080:8080 graft
            ```
            This command runs the Docker container and maps port 8080 of the container to port 8080 of the host machine. The application will be accessible at `http://localhost:8080/`.
    2. For Production:

        - Build the Docker image for local testing using the following command:
            ```bash
            docker build -t graft .
            ```
            This command sets the Flask environment to default production mode and builds the Docker image with the tag graft.
        - Run the Docker container using the following command:
            ```bash
            docker run -p 8080:8080 graft
            ```
            This command runs the Docker container and maps port 8080 of the container to port 8080 of the host machine. The application will be accessible at `http://localhost:8080/`.