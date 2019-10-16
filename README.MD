# generator.c

See [repo](https://github.com/WuLC/ViterbiAlgorithm/blob/master/Viterbi.py)

```python
states = ('Healthy', 'Fever')
 
observations = ('normal', 'cold', 'dizzy')
 
start_probability = {'Healthy': 0.9332, 'Fever': 0.0668}
 
transition_probability = {
   'Healthy' : {'Healthy': 0.3464, 'Fever': 0.6536},
   'Fever' :   {'Healthy': 0.0159, 'Fever': 0.9841},
   }
 
emission_probability = {
   'Healthy' : {'normal': 0.6308, 'cold': 0.0821, 'dizzy': 0.2871},
   'Fever'   : {'normal': 0.3430, 'cold': 0.2983, 'dizzy': 0.3587},
   }
```

## input
```asm
2 3 
```
2 -> "Healthy" and "Fever" <br >
3 -> "normal", "cold", "dizzy"
## output
```asm
2 3
0.9332 0.0668
0.3464 0.6536
0.0159 0.9841
0.6308 0.0821 0.2871
0.3430 0.2983 0.3587
```
