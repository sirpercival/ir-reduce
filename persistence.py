# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:14:08 2015

@author: gray
"""

#from kivy.storage.jsonstore import JsonStore
from kivy.storage import AbstractStore
import jsonpickle
#import demjson
#from json import dumps, load
import json
from os.path import exists
#from kivy.storage.dictstore import DictStore
import os, uuid, glob
#from datatypes import FitsImage
#jsonpickle.load_backend('demjson', 'encode', 'decode', 'JSONDecodeError')
#jsonpickle.set_preferred_backend('demjson')
#jsonpickle.set_preferred_backend('json')
#jsonpickle.handlers.register(FitsImage, jsonpickle.handlers.SimpleReduceHandler)

#class IRStore(JsonStore):
class IRStore(AbstractStore):
    '''Store implementation using jsonpickle to handle aribtrary python objects.'''
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self._data = {}
        self._is_changed = True
        super(IRStore, self).__init__(**kwargs)
    
    def __setitem__(self, key, values):
        self.put(key, values)
    
    def put(self, key, values):
        need_sync = self.store_put(key, values)
        if need_sync:
            self.store_sync()
        return need_sync
        
    def async_put(self, callback, key, **values):
        self._schedule(self.store_put_async,
                       key=key, value=values, callback=callback)
    
    def store_put_async(self, key, value, callback):
        try:
            value = self.put(key, value)
            callback(self, key, value)
        except:
            callback(self, key, None)
        
    def store_load(self):
        if not exists(self.filename):
            return
        with open(self.filename) as fd:
            data = fd.read()
            if len(data) == 0:
                return
            #self._data = demjson.decode(data)
            self._data = json.loads(data)
    
    def store_sync(self):
        if self._is_changed is False:
            return
        #demjson.encode_to_file(self.filename, self._data, overwrite=True)
        with open(self.filename, 'w') as fd:
            json.dump(self._data, fd)
    
    def store_exists(self, key):
        return key in self._data
    
    def store_get(self, key):
        return jsonpickle.decode(self._data[key])

    def store_put(self, key, value):
        self._data[key] = jsonpickle.encode(value)
        self._is_changed = True
        return True
    
    def store_delete(self, key):
        del self._data[key]
        self._is_changed = True
        return True
    
    def store_find(self, filters):
        for key, values in self._data.iteritems():
            found = True
            for fkey, fvalue in filters.iteritems():
                if fkey not in values:
                    found = False
                    break
                if values[fkey] != fvalue:
                    found = False
                    break
            if found:
                yield key, values

    def store_count(self):
        return len(self._data)

    def store_keys(self):
        return self._data.keys()
    
    

#instrumentdb = DictStore(os.path.join('storage','instrumentprofiles'))
#obsrundb = DictStore(os.path.join('storage','observingruns'))
#linelistdb = DictStore(os.path.join('storage','linelists'))
instrumentdb = IRStore(os.path.join('storage','instrumentprofiles.json'))
tracedir = lambda inst: instrumentdb[inst].tracedir
obsrundb = IRStore(os.path.join('storage','observingruns.json'))
linelistdb = IRStore(os.path.join('storage','linelists.json'))


#class AdHocDB(DictStore):
class AdHocDB(IRStore):
    def __init__(self, fname=None, **kw):
        self.fname = fname
        if not self.fname:
            while True:
                self.fname = os.path.join('storage/',str(uuid.uuid4())+'.json')
                #self.fname = os.path.join('storage',str(uuid.uuid4()))
                if not glob.glob(self.fname + '*'):
                    break
        super(AdHocDB, self).__init__(self.fname, **kw)
        #self.db = DictStore(self.fname)
        #self.db = IRStore(self.fname)
    
    #def __getitem__(self, key):
    #    return self.db[key]