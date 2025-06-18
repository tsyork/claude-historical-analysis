#!/bin/bash

# One-Command AWS Spot Fleet Transcription Deployment
# Usage: ./deploy-transcription.sh

set -e

echo "🚀 AWS Spot Fleet Transcription - One-Command Deploy"
echo "=================================================="

# Check required files
echo "📋 Checking required files..."
required_files=("spot-fleet-config.json" "interactive_transcribe.py" "credentials.json" "whisper-transcription-key.pem")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done
echo "✅ All required files present"

# Create working User Data script
echo "🔧 Creating automated setup script..."
cat > startup-script.sh << 'STARTUP'
#!/bin/bash

# Log everything to setup.log
exec > /home/ubuntu/setup.log 2>&1

echo "🚀 Starting automated transcription instance setup..."

# Update system and add Python 3.12 repository
echo "📦 Installing Python 3.12..."
apt update -y
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.12 python3.12-venv python3.12-dev

# Get pip for Python 3.12
echo "📥 Installing pip for Python 3.12..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create virtual environment as ubuntu user
echo "🔧 Creating virtual environment..."
cd /home/ubuntu
sudo -u ubuntu python3.12 -m venv transcription-env

# Install Python packages as ubuntu user
echo "📦 Installing Python packages..."
sudo -u ubuntu bash -c "
source /home/ubuntu/transcription-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper google-api-python-client google-auth google-cloud-storage tqdm certifi
"

# Install NVIDIA drivers
echo "🎮 Installing NVIDIA drivers..."
apt install -y ubuntu-drivers-common
ubuntu-drivers autoinstall

# Create quick-start script
cat > /home/ubuntu/quick-start.sh << 'QUICKSTART'
#!/bin/bash
cd /home/ubuntu

# Wait for files to be copied
echo "📂 Waiting for transcription files..."
while [ ! -f interactive_transcribe.py ] || [ ! -f credentials.json ]; do
    echo "   Waiting for files..."
    sleep 5
done

# Create auto-answers for Season 6, large-v3, auto-shutdown
echo -e "6\n1\n\n\nY\nY\nY" > auto-answers.txt

# Run transcription with auto-answers
source transcription-env/bin/activate
python interactive_transcribe.py < auto-answers.txt
QUICKSTART

chown ubuntu:ubuntu /home/ubuntu/quick-start.sh
chmod +x /home/ubuntu/quick-start.sh

# Create setup completion indicator
echo "ready" > /home/ubuntu/setup-complete.txt
chown ubuntu:ubuntu /home/ubuntu/setup-complete.txt

echo "✅ Automated setup complete! Rebooting to load NVIDIA drivers..."
sleep 30
reboot
STARTUP

# Convert to base64 for User Data
BASE64_SCRIPT=$(base64 -i startup-script.sh)

# Update spot fleet config with working User Data
sed -i.bak "s/\"UserData\": \".*\"/\"UserData\": \"$BASE64_SCRIPT\"/" spot-fleet-config.json

# Clean up temporary script
rm startup-script.sh

echo "✅ Updated deployment with working automation"

# Step 1: Launch Fleet
echo "🚀 Step 1: Launching Spot Fleet..."
FLEET_ID=$(aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config.json --query 'SpotFleetRequestId' --output text)
echo "Fleet ID: $FLEET_ID"

# Step 2: Wait for Instance
echo "⏳ Step 2: Waiting for instance to be ready..."
echo "This usually takes 2-3 minutes..."

while true; do
    STATUS=$(aws ec2 describe-spot-fleet-requests --spot-fleet-request-ids $FLEET_ID --query 'SpotFleetRequestConfigs[0].ActivityStatus' --output text)
    echo "   Status: $STATUS"
    
    if [ "$STATUS" = "fulfilled" ]; then
        echo "✅ Fleet is ready!"
        break
    elif [ "$STATUS" = "error" ]; then
        echo "❌ Fleet failed to launch. Checking error..."
        aws ec2 describe-spot-fleet-request-history --spot-fleet-request-id $FLEET_ID --start-time $(date -u -v-10M +%Y-%m-%dT%H:%M:%S)
        exit 1
    fi
    
    sleep 15
done

# Step 3: Get Instance Details
echo "📡 Step 3: Getting instance details..."
INSTANCE_ID=$(aws ec2 describe-spot-fleet-instances --spot-fleet-request-id $FLEET_ID --query 'ActiveInstances[0].InstanceId' --output text)
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"

# Step 4: Wait for Automated Setup
echo "⏳ Step 4: Waiting for automated setup to complete..."
echo "Installing Python 3.12, PyTorch, Whisper, and NVIDIA drivers..."
echo "This takes about 6-8 minutes total (including reboot)..."

# Wait for setup completion indicator
setup_complete=false
for i in {1..25}; do
    if ssh -i whisper-transcription-key.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f setup-complete.txt" 2>/dev/null; then
        echo "✅ Automated setup complete!"
        setup_complete=true
        break
    fi
    echo "   Setup in progress... ($i/25) - 30 seconds each"
    sleep 30
done

if [ "$setup_complete" = false ]; then
    echo "⚠️  Setup taking longer than expected. Continuing anyway..."
fi

# Wait for reboot to complete
echo "⏳ Waiting for instance to reboot and load NVIDIA drivers..."
sleep 90

# Wait for instance to come back online after reboot
echo "⏳ Waiting for instance to come back online..."
for i in {1..15}; do
    if ssh -i whisper-transcription-key.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "echo 'ready'" >/dev/null 2>&1; then
        echo "✅ Instance is back online!"
        break
    fi
    echo "   Waiting for reboot to complete... ($i/15)"
    sleep 15
done

# Step 5: Deploy Files
echo "📦 Step 5: Deploying transcription files..."
scp -i whisper-transcription-key.pem -o StrictHostKeyChecking=no interactive_transcribe.py ubuntu@$PUBLIC_IP:~/
scp -i whisper-transcription-key.pem -o StrictHostKeyChecking=no credentials.json ubuntu@$PUBLIC_IP:~/

echo "✅ Files deployed!"

# Step 6: Verify Setup
echo "🔍 Step 6: Verifying GPU and environment..."
ssh -i whisper-transcription-key.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "
    echo '📊 System Status:'
    echo 'GPU Status:'
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo 'GPU not ready yet'
    echo ''
    echo 'Python Environment:'
    source transcription-env/bin/activate
    python -c 'import torch; print(f\"PyTorch CUDA available: {torch.cuda.is_available()}\")'
    echo ''
    echo 'Files Ready:'
    ls -la interactive_transcribe.py credentials.json
"

# Step 7: Provide Next Steps
echo ""
echo "🎉 Deployment Complete!"
echo "=============================="
echo ""
echo "Your GPU transcription instance is ready! Here are your options:"
echo ""
echo "🎯 Option A: Interactive Transcription"
echo "   ssh -i whisper-transcription-key.pem ubuntu@$PUBLIC_IP"
echo "   source transcription-env/bin/activate"
echo "   python interactive_transcribe.py"
echo ""
echo "🚀 Option B: Quick Start (Season 6, large-v3, auto-shutdown)"
echo "   ssh -i whisper-transcription-key.pem ubuntu@$PUBLIC_IP './quick-start.sh'"
echo ""
echo "📊 Instance Info:"
echo "   Fleet ID: $FLEET_ID"
echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP: $PUBLIC_IP"
echo "   SSH Command: ssh -i whisper-transcription-key.pem ubuntu@$PUBLIC_IP"
echo ""
echo "🛑 To stop the fleet when done:"
echo "   aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids $FLEET_ID --terminate-instances"
echo ""
echo "💰 Current Cost: ~$0.20-0.30 per hour"
echo "⚡ Performance: ~8-10x faster than local processing"
echo ""

# Save fleet info for easy cleanup
echo $FLEET_ID > .current-fleet-id
echo $INSTANCE_ID > .current-instance-id  
echo $PUBLIC_IP > .current-instance-ip

echo "Fleet info saved to local files for easy management."
echo "Ready to transcribe! 🎙️"
