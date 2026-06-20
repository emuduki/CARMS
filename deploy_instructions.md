# Deploying CARMS to Microsoft Azure

This guide explains how to deploy CARMS to a Linux Virtual Machine on Azure (specifically a B1s VM, which is free under the **Azure for Students** subscription).

---

## Prerequisites
- An active Azure for Students account.
- SSH client (e.g. PowerShell, Git Bash, or Terminal).

---

## Step 1: Create a Linux Virtual Machine on Azure

1. Log in to the [Azure Portal](https://portal.azure.com/).
2. Search for **Virtual Machines** and click **Create** > **Azure virtual machine**.
3. Configure the VM with the following settings:
   - **Subscription**: `Azure for Students`
   - **Resource group**: Create new (e.g., `CARMS-RG`)
   - **Virtual machine name**: `carms-vm`
   - **Region**: Select a region close to you (e.g., `East US`)
   - **Image**: `Ubuntu Server 22.04 LTS - x64 Gen2`
   - **Size**: `Standard_B1s` (1 vcpu, 1 GiB memory — Free tier eligible!)
   - **Authentication type**: `SSH public key`
   - **Username**: `azureuser`
   - **SSH public key source**: Generate new key pair (or use existing if you have one).
4. Under **Inbound port rules**:
   - Allow selected ports: Check **SSH (22)**.
5. Click **Review + create**, then download the private key `.pem` file when prompted and create the VM.

---

## Step 2: Open Dash Web Dashboard Port (8050)

By default, Azure blocks all incoming traffic except SSH. We need to open port `8050` so you can access the dashboard.

1. In the Azure Portal, navigate to your newly created Virtual Machine `carms-vm`.
2. In the left menu under **Settings**, click on **Networking** (or **Network settings**).
3. Click **Add inbound port rule** and enter:
   - **Source**: `Any`
   - **Source port ranges**: `*`
   - **Destination**: `Any`
   - **Service**: `Custom`
   - **Destination port ranges**: `8050`
   - **Protocol**: `TCP`
   - **Action**: `Allow`
   - **Priority**: `310` (or default)
   - **Name**: `Port_8050_Dash`
4. Click **Add**.

---

## Step 3: Connect to the VM and Configure SWAP Space (CRITICAL)

Since the B1s VM only has 1GB of RAM, installing packages like PyTorch or running calculations will run out of memory (OOM). Allocating **SWAP space** provides extra virtual memory on the SSD.

1. Open your terminal/PowerShell and connect to your VM using your downloaded `.pem` key:
   ```bash
   ssh -i /path/to/your-key.pem azureuser@<YOUR_VM_PUBLIC_IP>
   ```
2. Run the following commands to create a 4GB Swap File:
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```
3. To make the swap file permanent across reboots, add it to `/etc/fstab`:
   ```bash
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
4. Verify the swap is active:
   ```bash
   free -h
   ```

---

## Step 4: Install Docker & Docker Compose on VM

### 4a. Install Docker Engine
```bash
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```
*Note: Log out and log back in (or run `newgrp docker`) for group permissions to take effect.*

### 4b. Install Docker Compose v2 (CLI Plugin)

> **Important**: `docker.io` from Ubuntu's apt repo does **not** include `docker-compose-plugin`. Install Compose v2 directly from GitHub as a CLI plugin instead:

```bash
mkdir -p ~/.docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/download/v2.27.1/docker-compose-linux-x86_64" \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
```

Verify the installation:
```bash
docker compose version   # Should print Docker Compose v2.x.x
```

---

## Step 5: Deploy the Code and Start CARMS

1. Clone your CARMS repository on the VM:
   ```bash
   git clone https://github.com/<your-username>/carms.git
   cd carms
   ```
2. Transfer or create your local config files:
   - Create `configs/config.local.yaml` with your private API keys (Binance, OANDA, FRED, etc.).
   ```bash
   nano configs/config.local.yaml
   ```
3. Build and launch the containerized application:
   ```bash
   docker compose up --build -d
   ```
4. Check the application logs to ensure it started successfully:
   ```bash
   docker compose logs -f
   ```

---

## Running Pipeline Phases

To execute data ingestion, training, or simulations inside the docker context:

- **Run Phase 1 (Data pipeline):**
  ```bash
  docker compose run --rm carms-cli python main.py --phase 1
  ```
- **Run Phase 2 (Train models):**
  ```bash
  docker compose run --rm carms-cli python main.py --phase 2
  ```

---

## Accessing the Dashboard

Once the dashboard container is running, open your web browser and navigate to:
```
http://<YOUR_VM_PUBLIC_IP>:8050
```

---

## Automated Deployment (CI/CD with GitHub Actions)

CARMS includes a GitHub Action workflow in `.github/workflows/deploy.yml` that automatically deploys your updates to the Azure VM every time you push to the `main` branch of your GitHub repository.

### Setting up Continuous Deployment

To enable this, follow these steps to add your server credentials to GitHub Repository Secrets:

1. **Get the required VM details**:
   - **VM Public IP**: `40.127.8.3`
   - **Username**: The VM user (typically `azureuser`)
   - **Private Key**: The content of the SSH private key `.pem` file you use to log in to the VM.

2. **Add Secrets to GitHub**:
   - Go to your repository on GitHub: `https://github.com/emuduki/CARMS`
   - Navigate to **Settings** (top tabs) > **Secrets and variables** (left sidebar) > **Actions**.
   - Click **New repository secret** for each of the following:
     - **Name**: `AZURE_VM_IP`
       - **Value**: `40.127.8.3`
     - **Name**: `AZURE_USERNAME`
       - **Value**: `azureuser` (or your actual VM ssh username)
     - **Name**: `AZURE_SSH_KEY`
       - **Value**: Paste the exact contents of your private key `.pem` file (make sure to include `-----BEGIN RSA PRIVATE KEY-----` / `-----BEGIN OPENSSH PRIVATE KEY-----` and the corresponding `-----END...` footer).

3. **Verify the Deployment**:
   - Commit and push your changes to your `main` branch.
   - Go to the **Actions** tab on your GitHub repository to watch the deployment run.
   - The workflow will securely SSH into your VM, run `git pull`, build the new Docker images, and restart the containers in the background.

---

## Troubleshooting

### `KeyError: 'ContainerConfig'`
This error occurs when using the legacy `docker-compose` v1.29.2 with Docker Engine v25+. Fix it by installing Docker Compose v2 as a CLI plugin:
```bash
mkdir -p ~/.docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/download/v2.27.1/docker-compose-linux-x86_64" \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
# Verify, then use 'docker compose' (no hyphen) for all commands
docker compose version
```
> Note: `docker-compose-plugin` via `apt` is only available when using Docker's **official** repo, not Ubuntu's `docker.io` package.
