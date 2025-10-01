Please read before running the code.

Install all packages used:
pip install -r requirements.txt


To run SFDA experiments:


Office-31 or Office-Home

Step 1. Source representation training: 
python source_pretrain_office.py

Step 2: Move the pre-trained models from the folder "san" to the folder "weight".

Step 3: Diffusion model learning:
python diffusion_office.py

Step 4: Target adaptation
python adaptation_office_with_diffusion.py


VisDA

Step 1. Source representation training: 
python source_pretrain_visda.py

Step 2: Move the pre-trained models from the folder "san" to the folder "weight".

Step 3: Diffusion model learning:
python diffusion_visda.py

Step 4: Target adaptation
python adaptation_visda_with_diffusion.py