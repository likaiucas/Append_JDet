# Append_JDet
add new functions for jdet pkg

step1: install your Jittor and JDet
```shell
git clone https://github.com/Jittor/JDet
cd JDet
python -m pip install -r requirements.txt

# suggest this 
python setup.py develop
```

step2: install my packages
```shell
cd ..
git clone https://github.com/CoolbreezeKevin/Append_JDet.git

mv Append_JDet/fair1M_label_to_fair1M_5.py JDet/python/jdet/data/devkits
mv Append_JDet/fair_preprocess_config_1024.py JDet/configs/preprocess
mv Append_JDet/fpn.py JDet/python/jdet/models/necks
mv Append_JDet/preprocess_copy.py JDet/tools
mv Append_JDet/fuse_datasets.py JDet/python/jdet/tools
mv Append_JDet/trainsforms.py JDet/python/jdet/data/devkits/transforms.py
mv Append_JDet/swa_model.py JDet/tools
```
