# mars-3-months
This is the code used to calculate estimates for various important things for Starship Mars mission described at:https://www.nature.com/articles/s41598-025-00565-7. I should organize it properly at some point. The main files of note are hyperbolic-earth-entry.ipynb and general_trajectory.ipynb in the trajectories folder. The general trajectory file is a generic lambert solver for any planet, calculating departure and arrival C3, along with optional de-acceleration burn upon arrival. 


Python details (note I upgraded to python 3.10 later on, not all the code has been checked in 3.10):
python 3.10
pip install "numpy==1.22.0" "scipy==1.7.3" "astropy==5.1" "numba==0.58.0"
pip install --no-deps "astroquery==0.4.5"
pip install --no-deps "poliastro==0.17.0"
pip install "requests==2.28.2" "keyring==23.13.1" "beautifulsoup4==4.12.2" "html5lib==1.1" "pyvo==1.5.1"
pip install "jplephem==2.18" "matplotlib==3.5.3" "pandas==1.4.4" "plotly==5.17.0"
