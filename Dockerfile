# Step 1: Use an official base image
FROM python:3.11.9-slim

# Step 2: Install C++ compiler and other dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    git \
    && apt-get clean

# Step 3: Set the working directory
WORKDIR /app

# Step 4: Copy the source code into the container
COPY . /app

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Compile the source code
RUN make

# Step 7: Run The APP
CMD ["python3","webui.py"]