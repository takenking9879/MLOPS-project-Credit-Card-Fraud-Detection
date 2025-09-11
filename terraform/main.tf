terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 3.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.0"
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

# modules
module "ecr" {
  source   = "./modules/ecr"
  ecr_repo = var.ecr_repo
}

module "sg" {
  source   = "./modules/security_group"
  app_port = var.app_port
}

module "keypair" {
  source   = "./modules/keypair"
  key_name = var.key_name
}

module "ec2" {
  source         = "./modules/ec2"
  ami_id         = var.ami_id
  instance_type  = var.instance_type
  key_name       = module.keypair.key_name
  security_group = module.sg.sg_id
  subnet_id      = data.aws_subnets.default.ids[0]
  runner_token   = var.runner_token
  app_port       = var.app_port
}
