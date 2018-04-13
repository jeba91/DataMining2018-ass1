# sort the dataframe
dataset.sort_values(by=['variable'])
print(dataset)
# set the index to be this and don't drop
dataset.set_index(keys=['variable'], drop=False,inplace=True)
print(dataset)

        data.sort_values(by=['id'])
        # set the index to be this and don't drop
        data.set_index(keys=['id'], drop=False, inplace=True)
        # get a list of id names
        # Total nummer of id's = 27 (some are missing!)
