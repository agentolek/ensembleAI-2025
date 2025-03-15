# Findings:
- current acc on train data is 25%, that means model spits out random stuff
- based on that i guess i should search for patterns in data provided

# Ideas:
- silency maps of where model looks
- grad-cam
- try adding some noise, if the models performs better train a new one on that data (we might loose some acc)

# UPDATE
The dataset is clean, just train new model