---
annotations_creators:
- machine-generated
language:
- en
language_creators:
- machine-generated
license: mit
multilinguality:
- monolingual
pretty_name: TOFU
size_categories:
- 1K<n<10K
source_datasets:
- original
tags:
- unlearning
- question answering
- TOFU
- NLP
- LLM
task_categories:
- question-answering
task_ids:
- closed-domain-qa
configs:
- config_name: full
  data_files: full.json
  default: true
- config_name: forget01
  data_files: forget01.json
- config_name: forget05
  data_files: forget05.json
- config_name: forget10
  data_files: forget10.json
- config_name: forget20
  data_files: forget20.json
- config_name: forget30
  data_files: forget30.json
- config_name: forget50
  data_files: forget50.json
- config_name: forget90
  data_files: forget90.json
- config_name: retain90
  data_files: retain90.json
- config_name: retain95
  data_files: retain95.json
- config_name: retain99
  data_files: retain99.json
- config_name: world_facts
  data_files: world_facts.json
- config_name: real_authors
  data_files: real_authors.json
- config_name: forget01_perturbed
  data_files: forget01_perturbed.json
- config_name: forget05_perturbed
  data_files: forget05_perturbed.json
- config_name: forget10_perturbed
  data_files: forget10_perturbed.json
- config_name: retain_perturbed
  data_files: retain_perturbed.json
- config_name: world_facts_perturbed
  data_files: world_facts_perturbed.json
- config_name: real_authors_perturbed
  data_files: real_authors_perturbed.json
- config_name: full_minus_forget01
  data_files: full_minus_forget01.json
- config_name: full_minus_forget05
  data_files: full_minus_forget05.json
- config_name: full_minus_forget10
  data_files: full_minus_forget10.json
- config_name: full_minus_forget20
  data_files: full_minus_forget20.json
- config_name: full_minus_forget30
  data_files: full_minus_forget30.json
- config_name: full_minus_forget50
  data_files: full_minus_forget50.json
- config_name: full_minus_forget90
  data_files: full_minus_forget90.json


