terraform {
required_version = ">= 1.0"
required_providers {
aws = {
source = "hashicorp/aws"
version = "~> 5.0"
}
tls = {
source = "hashicorp/tls"
version = "~> 4.0"
}
local = {
source = "hashicorp/local"
version = "~> 2.0"
}
}
}

provider "aws" {
region = var.region
}

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
  description = "Security group for the app"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "App port"
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
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

  tags = {
    Name = var.sec_group
  }
}

// Generamos una llave privada (PEM) localmente y registramos la public key en AWS
resource "tls_private_key" "key" {
algorithm = "RSA"
rsa_bits = 4096
}

resource "aws_key_pair" "deployer" {
key_name = var.key_name
public_key = tls_private_key.key.public_key_openssh
}

resource "local_file" "private_key" {
content = tls_private_key.key.private_key_pem
filename = "${path.module}/${var.key_name}.pem"
file_permission = "0400"
}

resource "aws_ecr_repository" "repo" {
  name = var.ecr_repo
}

resource "aws_instance" "app" {
ami = var.ami_id
instance_type = var.instance_type
subnet_id = element(data.aws_subnets.default.ids, 0)
vpc_security_group_ids = [aws_security_group.app_sg.id]
key_name = aws_key_pair.deployer.key_name
tags = {
Name = "my-app-instance"
}


root_block_device {
volume_size = 30
}

# user_data replica el script de provisioning que tenías por SSH
user_data = <<-EOF
#!/bin/bash
set -e


sudo apt-get update -y
sudo apt-get upgrade -y


# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker


# Crear carpeta del runner como ubuntu
mkdir -p actions-runner && cd actions-runner
chown ubuntu:ubuntu /home/ubuntu/actions-runner
curl -o actions-runner-linux-x64-2.328.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.328.0/actions-runner-linux-x64-2.328.0.tar.gz
echo "01066fad3a2893e63e6ca880ae3a1fad5bf9329d60e77ee15f2b97c148c3cd4e  actions-runner-linux-x64-2.328.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.328.0.tar.gz

./config.sh --url ${var.repo_url} \
 --token ${var.token} \
 --name self-hosted \
 --unattended
./run.sh &
EOF
}