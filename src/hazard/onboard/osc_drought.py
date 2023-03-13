from datetime import datetime
import os, sys
import itertools
import numpy as np
import xarray as xr
import s3fs
import xclim


class OscSpei:

    def __init__(
            self,
            write_folder,
            gcm="ACCESS-CM2",
            scenario="ssp585",
            years=np.arange(1950,1952),
            calib_start=datetime(1950,1,1),
            calib_end=datetime(1952,1,1),
            freq = "MS",
            window = 12,
            dist = "gamma",
            method = "APP",
            lat_min = -60.0,
            lat_max = 90.0,
            lon_min = 0.0,
            lon_max = 360.0,
            lat_delta = 10.0,
            lon_delta = 10.0
        ):
        self.write_folder = write_folder
        self.gcm = gcm
        self.scenario = scenario
        self.years = years
        self.calib_start = calib_start
        self.calib_end = calib_end
        self.freq = freq
        self.window = window
        self.dist = dist
        self.method = method
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_delta = lat_delta
        self.lon_delta = lon_delta
        self.input_variables = ['pr']#['tas','pr']
        self.datasource = NexGddpCmip6()
        self.fs = s3fs.S3FileSystem(anon=True)

    def get_datachunks(self):
        lat_bins = np.arange(self.lat_min,self.lat_max + 0.1*self.lat_delta,self.lat_delta)
        lon_bins = np.arange(self.lon_min,self.lon_max + 0.1*self.lon_delta,self.lon_delta)
        data_chunks = {str(i).zfill(4): dict(list(d[0].items())+list(d[1].items())) for i,d in enumerate(itertools.product([{'lat_min':x[0],'lat_max':x[1]} for x in zip(lat_bins[:-1],lat_bins[1:])],[{'lon_min':x[0],'lon_max':x[1]} for x in zip(lon_bins[:-1],lon_bins[1:])]))}
        data_chunks = {'0361' : data_chunks['0361']}
        return data_chunks
    
    def get_chunk_filename(self,variable,data_chunk_name):
        return os.path.join(self.write_folder, self.gcm+"_"+self.scenario+"_"+variable+"_"+data_chunk_name+".nc")
    
    def get_dataset(self,input_variable,year):
        scenario_ = "historical" if year < 2015 else self.scenario
        datapath,_ = self.datasource.path(gcm=self.gcm, scenario=scenario_, quantity=input_variable, year=year)
        f = self.fs.open(datapath, 'rb')
        ds = xr.open_dataset(f).astype('float32').compute()
        return ds
    
    def rechunk_dataset(self):
        data_chunks = self.get_datachunks()
        for input_variable in self.input_variables:
            for year in self.years:
                ds = self.get_dataset(input_variable,year)
                for data_chunk in data_chunks:
                    chunk_limits = data_chunks[data_chunk]
                    chunk_filename = self.get_chunk_filename(input_variable,data_chunk)
                    ds_chunk_curr = ds.sel(lat=slice(chunk_limits['lat_min'],chunk_limits['lat_max']),lon=slice(chunk_limits['lon_min'],chunk_limits['lon_max']))
                    if not(os.path.exists(chunk_filename)):
                        ds_chunk = ds_chunk_curr
                    else:
                        ds_chunk_exis = xr.load_dataset(chunk_filename)
                        ds_chunk = xr.concat([ds_chunk_exis,ds_chunk_curr],"time")
                    ds_chunk.to_netcdf(chunk_filename)

    def create_pet_datasets(self):
        data_chunks = self.get_datachunks()
        for data_chunk in data_chunks:
            tas_chunk_filename = self.get_chunk_filename("tas",data_chunk)
            pet_chunk_filename = self.get_chunk_filename("pet",data_chunk)
            ds_tas = xr.open_dataset(tas_chunk_filename)
            ds_pet = xclim.indices.potential_evapotranspiration(tas=ds_tas['tas'],method='MB05').astype('float32').compute().to_dataset(name='pet')
            ds_pet.to_netcdf(pet_chunk_filename)

    def create_spei_datasets(self):
        data_chunks = self.get_datachunks()
        for data_chunk in data_chunks:
            pet_chunk_filename = self.get_chunk_filename("pet",data_chunk)
            pr_chunk_filename = self.get_chunk_filename("pr",data_chunk)
            spei_chunk_filename = self.get_chunk_filename("spei"+"_"+self.freq+"_"+str(self.window).zfill(2)+"_"+self.dist+"_"+self.method, data_chunk)
            ds_pet = xr.open_dataset(pet_chunk_filename)
            ds_pr = xr.open_dataset(pr_chunk_filename)
            da_wb = ds_pr['pr'] - ds_pet['pet']
            da_wb.attrs['units'] = ds_pr['pr'].attrs['units']
            da_wb_calib = da_wb.sel(time=slice(self.calib_start,self.calib_end))
            ds_spei = xclim.indices.standardized_precipitation_evapotranspiration_index(
                da_wb,
                da_wb_calib,
                freq=self.freq,
                window=self.window,
                dist=self.dist,
                method=self.method
            ).astype('float32').to_dataset(name='spei').resample(time="1MS").mean(dim='time').compute() # ,encoding={'spei':{'dtype':'float32'},'lat':{'dtype':'float32'},'lon':{'dtype':'float32'},'time':{'dtype':'int64'}}
            ds_spei.to_netcdf(spei_chunk_filename,engine='netcdf4')

    def run_spei_calculation_job(self):
        #self.rechunk_dataset()
        #self.create_pet_datasets()
        self.create_spei_datasets()
        print("Done")


if __name__ == "__main__":

    sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2]))
    from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6

    write_folder = r"C:\Temp"
    osc_spei = OscSpei(write_folder=write_folder)
    osc_spei.run_spei_calculation_job()
    print("Completed!")
