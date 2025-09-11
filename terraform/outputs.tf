output "ecr_uri" {
  description = "ECR repository URI"
  value       = module.ecr.ecr_uri
}

output "ec2_public_ip" {
  description = "Public IPv4 of EC2 instance"
  value       = module.ec2.public_ip
}
