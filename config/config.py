"ICI test set: Hellmann cohort"
class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        self.dataset = 'gs'
        self.model = 'gcn'  
        self.learning_rate = 0.0005  
        self.epochs  = 1000                    
        self.hidden1 = 200 
        self.dropout = 0.0  
        self.early_stopping = 10  
        self.seed0 = 190  
        self.seed = 139    
        self.size = 468
        self.test_set_name = 'Hellmann'
                        
# "ICI test set: Liu cohort"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn'  
#         self.learning_rate = 0.05           
#         self.epochs  = 1000          
#         self.hidden1 = 200  
#         self.dropout = 0.0  
#         self.early_stopping = 100 
#         self.seed0 = 42  
#         self.seed =  59   
#         self.size = 468
#         self.test_set_name = 'Liu'

# "TCGA test set: UCEC"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn' 
#         self.learning_rate = 0.05 
#         self.epochs  = 1000
#         self.hidden1 = 200  
#         self.dropout = 0.0  
#         self.early_stopping = 10 
#         self.seed0 = 184  
#         self.seed = 72                        
#         self.size = 468
#         self.test_set_name = 'TCGA-UCEC'  

# "TCGA test set: COAD"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn'  
#         self.learning_rate = 0.05   
#         self.epochs  = 1000
#         self.hidden1 = 200 
#         self.dropout = 0.0 
#         self.early_stopping = 10 
#         self.seed0 = 184
#         self.seed = 72                           
#         self.size = 468
#         self.test_set_name = 'TCGA-COAD'  

# "TCGA test set: BRCA"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn'  
#         self.learning_rate = 0.05   
#         self.epochs  = 1000
#         self.hidden1 = 200  
#         self.dropout = 0.0  
#         self.early_stopping = 10 
#         self.seed0 = 184
#         self.seed = 72
#         self.size = 468
#         self.test_set_name = 'TCGA-BRCA'   

# "TCGA test set: LUSC"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn'  
#         self.learning_rate = 0.05   
#         self.epochs  = 1000
#         self.hidden1 = 200  
#         self.dropout = 0.0  
#         self.early_stopping = 10
#         self.seed0 = 184
#         self.seed = 72                         
#         self.size = 468
#         self.test_set_name = 'TCGA-LUSC' 

# "TCGA test set: BLCA"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn'  
#         self.learning_rate = 0.05  
#         self.epochs  = 1000
#         self.hidden1 = 200 
#         self.dropout = 0.0 
#         self.early_stopping = 10
#         self.seed0 = 184
#         self.seed = 72                            
#         self.size = 468
#         self.test_set_name = 'TCGA-BLCA' 

# "TCGA test set: SKCM"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn'  
#         self.learning_rate = 0.05
#         self.epochs  = 1000
#         self.hidden1 = 200  
#         self.dropout = 0.0 
#         self.early_stopping = 10 
#         self.seed0 = 184
#         self.seed = 77
#         self.size = 468
#         self.test_set_name = 'TCGA-SKCM'     
        
# "TCGA test set: CESC"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn' 
#         self.learning_rate = 0.05  
#         self.epochs  = 1000
#         self.hidden1 = 200
#         self.dropout = 0.0 
#         self.early_stopping = 20 
#         self.seed0 = 184
#         self.seed = 72                            
#         self.size = 468
#         self.test_set_name = 'TCGA-CESC'  

# "TCGA test set: LUAD"
# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
#         self.dataset = 'gs'
#         self.model = 'gcn' 
#         self.learning_rate = 0.05   
#         self.epochs  = 1000
#         self.hidden1 = 200  
#         self.dropout = 0.0  
#         self.early_stopping = 10 
#         self.seed0 = 184
#         self.seed = 72                          
#         self.size = 468
#         self.test_set_name = 'TCGA-LUAD' 
