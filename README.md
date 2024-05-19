The source code of paper "VBF Net: Boundary Completion in Full Waveform Inversion using Fusion Networks".  


Steps:  
1. Create these two folders under "VBF-dev":  
model_ws # models saved  
results  # results saved  

2. pip install scipy  
pip install opencv-python  
pip install lpips  

3. Configure "config.yml"  

4. Run "VBF.py"  


Example:  
We give a sample of CurveFaultA dataset and its trained models under this link:  
Link: https://pan.baidu.com/s/1bb739LDQww2Wt0XSd3-ORg  
Extraction code: ca8y  

Copy them under this project.  

Configure "config.yml" with setting:  
1) DATASET_PATH: 'datasets/CurveFaultA'  
2) MODEL_TYPE as you desired (train or test)  
3) Uncomment "OpenFWI datasets"'s configuration code  

Run "VBF.py" with:  
1) model.train()  
2) model.test(testID=37) # Generate the velocity model by conducting predictions on the 37th seismic test data. The velocity model file is then saved in the "results" folder.  



