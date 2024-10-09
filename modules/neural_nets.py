import numpy as np
from modules import data_processing as dp

class Models():
    
    def __init__(self):
        self.t_min = 0
        self.t_max = 0
        
        self.model_100_400 = self._load("model_100_400")
        self.model_200_500 = self._load("model_200_500")
        self.model_300_600 = self._load("model_300_600")
        self.model_300_800 = self._load("model_300_800")
        self.model_400_700 = self._load("model_400_700")

    def _select_model(self, temp_array):
        t_min_data = temp_array[0]
        t_max_data = temp_array[-1]
        
        conditions = [
            (300, 800, self.model_300_800),
            (400, 700, self.model_400_700),
            (300, 600, self.model_300_600),
            (200, 500, self.model_200_500),
            (100, 400, self.model_100_400)
        ]
        
        for t_min_condition, t_max_condition, model in conditions:
            if t_min_data <= t_min_condition and t_max_data >= t_max_condition:
                print("==========")
                print(f"Return {t_min_condition}-{t_max_condition} model since t_min = {t_min_data} and t_max = {t_max_data}")
                print("==========")

                self.t_min = t_min_condition
                self.t_max = t_max_condition
                
                self.current_model = model
                return
                
        print(f"Return no model since t_min = {t_min_data} and t_max = {t_max_data}")
        self.current_model = None
            
    def get_prediction(self, temp_array, see_array):
        self._select_model(temp_array)
        if (self.current_model != None):
            see_array = self._interpolate_seebeck(temp_array, see_array)
            trans_see_array = self._transform_seebeck(see_array)
            pred = self.current_model.predict(trans_see_array.reshape(1,-1))
            
            return self._transform_prediction(see_array, self._reverse_norm(pred[0]))
        else:
            return None

    def _load(self, model_name):
        from tensorflow.keras.models import model_from_json
        
        with open(f"Models/{model_name}_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(f"Models/{model_name}_weights.h5")

        return loaded_model

    def _transform_seebeck(self, see_array):
        if (np.mean(see_array[:3]) > 0):
            return see_array*1e3
        else:
            return (-1)*see_array*1e3
        
    def _transform_prediction(self, see_array, prediction):  
        if (np.mean(see_array[:3]) > 0):
            return prediction
        else:
            mass = 1/prediction[0]
            gap = prediction[1]
            fermi_energy = gap - prediction[2]
            return [mass, gap, fermi_energy]
        
    def _interpolate_seebeck(self, temp_array, see_array):
        inter_data = dp.interpolate_data(dp.fit_data(np.column_stack((temp_array, see_array))), np.linspace(self.t_min, self.t_max, 7))
        
        return inter_data[:,1]

    def _reverse_norm(self, pred):
        mass_pred = self._inverse_softsign(pred[0], 0, 1)
        gap_pred = self._inverse_softsign(pred[1], 2000, 1000)
        fermiEnergy_pred = self._inverse_softsign(pred[2], -500, 1000)
        
        return [mass_pred, gap_pred, fermiEnergy_pred]

    def _inverse_softsign(self, y, shift, factor):
        return (shift+(factor-np.sign(y)*shift)*y)/(1-np.abs(y))