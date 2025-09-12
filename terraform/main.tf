provider "aws" {
  region = var.region
}

# -------------------------
# ECR
# -------------------------
resource "aws_ecr_repository" "mlops_repo" {
  name = var.ecr_repo
}

# -------------------------
# Security Group
# -------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "app_sg" {
  name        = var.sec_group
  description = "Security group for app"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -------------------------
# Key Pair
# -------------------------
resource "tls_private_key" "example" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "my_key" {
  key_name   = var.key_name
  public_key = tls_private_key.example.public_key_openssh
}

# -------------------------
# EC2 Instance
# -------------------------
resource "aws_instance" "app_instance" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.app_sg.id]
  key_name               = aws_key_pair.my_key.key_name

  tags = {
    Name = "my-app-instance"
  }

  root_block_device {
    volume_size = 30
  }

  user_data = <<-EOF
              #!/bin/bash
              set -e
              sudo apt-get update -y
              sudo apt-get upgrade -y
              curl -fsSL https://get.docker.com -o get-docker.sh
              sudo sh get-docker.sh
              sudo usermod -aG docker ubuntu
              newgrp docker

              mkdir -p /home/ubuntu/actions-runner && cd /home/ubuntu/actions-runner
              curl -o actions-runner-linux-x64-2.328.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.328.0/actions-runner-linux-x64-2.328.0.tar.gz
              echo "01066fad3a2893e63e6ca880ae3a1fad5bf9329d60e77ee15f2b97c148c3cd4e  actions-runner-linux-x64-2.328.0.tar.gz" | shasum -a 256 -c
              tar xzf ./actions-runner-linux-x64-2.328.0.tar.gz

              ./config.sh --url https://github.com/takenking9879/MLOPS-project-Credit-Card-Fraud-Detection \
                          --token ${var.token} \
                          --name self-hosted \
                          --unattended

              ./run.sh &
              EOF
}

# -------------------------
# Outputs
# -------------------------
output "public_ip" {
  description = "Public IP of EC2 instance"
  value       = aws_instance.app_instance.public_ip
}

output "ecr_repository_url" {
  description = "ECR repository URI"
  value       = aws_ecr_repository.mlops_repo.repository_url
}
