
import os
import dictionary_driver

class Dictionary_Manager() :

    def __init__( self, folderPath, taskName ) :

        self.taskName = taskName
        self.dictionary_driver = dictionary_driver.Dictionary_Driver()

        self.dictionaries_path = os.path.join(folderPath, 'task' + '_' + self.taskName + '.xlsx' )
        # self.dictionaries_backup_path = os.path.join( config['history'], 'Backup Dictionaries' +  '_' + self.taskName + '.xlsx' )

        self.dictionary_driver.Ensure_File_Sheet_Exists( self.dictionaries_path, sheetName = 'strings' )


    def Save_Dictionaries( self, folderPath, filename = None, dictionaries = {}, overwrite = True, horizontal = False, list_mode = False ) :

        path = None
        if filename is not None : path = os.path.join( folderPath, filename + ".xlsx" )
        else : path = self.dictionaries_path

        try :
            for dict_name, dict_body in dictionaries.items() :
                self.dictionary_driver.Ensure_File_Sheet_Exists( path, sheetName = dict_name )
                self.dictionary_driver.Save_Data_Dict( path, sheetName = dict_name, dictionary = dict_body, overwrite = overwrite, horizontal = horizontal, list_mode = list_mode )
        except :
            raise Exception( "Failture saving dictionaries")

        # if path == self.dictionaries_path : # Central history is managed by this obejct.
        #     self.dictionary_driver.Backup( self.dictionaries_path, self.dictionaries_backup_path )

        return
    

    def Load_Dictionaries( self, folderPath, filename = None, dict_names = [], reduce = 'last', horizontal = False, list_mode = False  ) :

        path = None
        if filename is not None : path = os.path.join( folderPath, filename + ".xlsx" )
        else : path = self.dictionaries_path

        dictionaries = {}

        try :
            for dict_name in dict_names :
                self.dictionary_driver.Ensure_File_Sheet_Exists( path, sheetName = dict_name )
                dictionaries[ dict_name ] = self.dictionary_driver.Load_Data_Dict( path, sheetName = dict_name, horizontal = horizontal, reduce = reduce, list_mode = list_mode )
        except :
            raise Exception( "Failure loading dintionaries from {}".format( path ) )

        return dictionaries
