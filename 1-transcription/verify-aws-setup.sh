#!/bin/bash

echo "🔍 Verifying AWS CLI Setup for Spot Fleet Transcription"
echo "======================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to check command and show result
check_command() {
    local description="$1"
    local command="$2"
    local expected_pattern="$3"
    
    echo -n "Checking $description... "
    
    if result=$(eval "$command" 2>&1); then
        if [[ -z "$expected_pattern" ]] || echo "$result" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✅ PASS${NC}"
            return 0
        else
            echo -e "${RED}❌ FAIL${NC}"
            echo "   Expected: $expected_pattern"
            echo "   Got: $result"
            return 1
        fi
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "   Error: $result"
        return 1
    fi
}

echo ""
echo "1. Basic AWS CLI Configuration"
echo "------------------------------"

check_command "AWS CLI version" "aws --version" "aws-cli"
check_command "AWS credentials" "aws sts get-caller-identity" "Account"
check_command "Default region" "aws configure get region" "us-"

echo ""
echo "2. Required AWS Permissions"
echo "---------------------------"

check_command "EC2 access" "aws ec2 describe-regions --max-items 1" "Regions"
check_command "IAM access" "aws iam list-roles --max-items 1" "Roles"
check_command "Spot pricing access" "aws ec2 describe-spot-price-history --instance-types g4dn.xlarge --max-items 1" "SpotPrices"

echo ""
echo "3. Required AWS Resources"
echo "-------------------------"

check_command "Default VPC" "aws ec2 describe-vpcs --filters 'Name=is-default,Values=true'" "VpcId"
check_command "Available subnets" "aws ec2 describe-subnets --filters 'Name=default-for-az,Values=true'" "SubnetId"
check_command "Spot Fleet service role" "aws iam get-role --role-name AWSServiceRoleForEC2SpotFleet" "RoleName"

echo ""
echo "4. SSH Key Pair"
echo "---------------"

if [ -f "whisper-transcription-key.pem" ]; then
    echo -e "SSH key file... ${GREEN}✅ FOUND${NC}"
    
    # Check key permissions
    perms=$(stat -f "%OLp" whisper-transcription-key.pem 2>/dev/null || stat -c "%a" whisper-transcription-key.pem 2>/dev/null)
    if [ "$perms" = "400" ]; then
        echo -e "SSH key permissions... ${GREEN}✅ CORRECT (400)${NC}"
    else
        echo -e "SSH key permissions... ${YELLOW}⚠️  NEEDS FIXING${NC}"
        echo "   Run: chmod 400 whisper-transcription-key.pem"
    fi
else
    echo -e "SSH key file... ${RED}❌ MISSING${NC}"
    echo "   Run: aws ec2 create-key-pair --key-name whisper-transcription-key --query 'KeyMaterial' --output text > whisper-transcription-key.pem"
fi

echo ""
echo "5. G4dn Instance Availability"
echo "-----------------------------"

check_command "G4dn pricing data" "aws ec2 describe-spot-price-history --instance-types g4dn.xlarge --max-items 1" "g4dn.xlarge"

# Check availability in multiple AZs
echo -n "G4dn availability zones... "
az_count=$(aws ec2 describe-availability-zones --query 'length(AvailabilityZones)' --output text)
if [ "$az_count" -ge 2 ]; then
    echo -e "${GREEN}✅ $az_count AZs available${NC}"
else
    echo -e "${YELLOW}⚠️  Only $az_count AZ available${NC}"
fi

echo ""
echo "6. Account Limits"
echo "-----------------"

# Check EC2 limits
echo -n "EC2 instance limits... "
if limit_info=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08 2>/dev/null); then
    limit=$(echo "$limit_info" | grep -o '"Value": [0-9]*' | cut -d' ' -f2)
    if [ "$limit" -ge 8 ]; then
        echo -e "${GREEN}✅ $limit vCPUs available${NC}"
    else
        echo -e "${YELLOW}⚠️  Only $limit vCPUs (may need increase)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Could not check limits${NC}"
fi

echo ""
echo "📋 Setup Summary"
echo "=================="

echo ""
echo "If all checks passed, you're ready to run:"
echo "  ./start-transcription-fleet.sh"
echo ""
echo "If there were failures, resolve them first:"
echo "  - Install/configure AWS CLI"
echo "  - Set correct IAM permissions"
echo "  - Create required resources"
echo ""
echo "💡 For help with failures, check:"
echo "  - AWS CLI documentation: https://docs.aws.amazon.com/cli/"
echo "  - IAM permissions: https://docs.aws.amazon.com/IAM/"