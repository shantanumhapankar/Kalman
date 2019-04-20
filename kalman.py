import numpy as np
from geopy import distance
import math

# Earths gravity
GRAVITY = 9.80665
# Radius in meters
EARTH_RADIUS = 6371 * 1000.0 

# List of all data_dicts
data_list = []
data_dict = {
    "timestamp": 0,
    "gps_lat": 0,
    "gps_lon": 0,
    "gps_alt": 0,
    "pitch": 0,
    "yaw": 0,
    "roll": 0,
    "rel_forward_acc": 0,
    "rel_up_acc": 0,
    "abs_north_acc": 0,
    "abs_east_acc": 0,
    "abs_up_acc": 0
}

# Convert DMS to degrees.
def parseDms(lat, latdir, lon, londir):
    deg = int(lat/100)
    seconds = lat - (deg * 100)
    latdec  = deg + (seconds/60)
    if latdir == 'S': latdec = latdec * -1
    deg = int(lon/100)
    seconds = lon - (deg * 100)
    londec  = deg + (seconds/60)
    if londir == 'W': londec = londec * -1
    return latdec, londec

def degreeToRadians(deg):
    return deg * math.pi / 180.0

def RadiansToDegrees(rad):
    return rad * 180.0 / math.pi

def getDistanceM(lat1, lon1, lat2, lon2):
    dlon = (lon2 - lon1) * math.pi / 180.0
    dlat = (lat2 - lat1) * math.pi / 180.0
    a = math.pow(math.sin(dlat/2.0), 2) + math.cos(lat1 * math.pi / 180.0) * math.cos(lat2 * math.pi / 180.0) * math.pow(math.sin(dlon/2.0), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    return EARTH_RADIUS * c

# Convert lat and long points from degrees to meters using the haversine formula. 
def LatLonToM(lat, lon):
    # latdis = distance.great_circle((lat,0.0),(0.0,0.0)).meters
    # londis = distance.great_circle((0.0,lon),(0.0,0.0)).meters
    latdis = getDistanceM(lat, 0.0, 0.0, 0.0)
    londis = getDistanceM(0.0, lon, 0.0, 0.0)
    if lat < 0: latdis *= -1
    if lon < 0: londis *= -1
    return latdis,londis

# Based on the current lat, long, calculate point ahead at distance 'dist' and angle 'azimuth'
def getPointAhead(lat, lon, dist, azimuth):
    frac = float(dist / EARTH_RADIUS)

    bearing = degreeToRadians(azimuth)

    lat = degreeToRadians(lat)
    lon = degreeToRadians(lon)

    lat1 = math.asin( math.sin(lat) * math.cos(frac) + math.cos(lat) * math.sin(frac) * math.cos(bearing))

    lon1 = lon + math.atan2( math.sin(bearing) * math.sin(frac) * math.cos(lat), math.cos(frac) - (math.sin(lat) * math.sin(lat1)))
    lon1 = ((lon1+3*math.pi) % (2*math.pi)) - math.pi

    return (RadiansToDegrees(lat1), RadiansToDegrees(lon1))

def metersToGeopoint(latM, lonM):
    lat, lon = 0.0, 0.0
    # Get point at East
    lat, lon = getPointAhead(lat, lon, lonM, 90.0)
    # Get point at NE
    lat, lon = getPointAhead(lat, lon, latM, 0.0)
    return lat, lon

def readData():
    with open('data.txt', 'r') as f:
        text = f.read().splitlines()
        del text[-1]
        for line in text:
            temp = line.replace(' -> ',',').split(',')
            time = temp[0].split(':')
            data_dict['timestamp'] = float(time[0])*3600 + float(time[1])*60 + float(time[2])
            lat, lon = parseDms(float(temp[1]), temp[2], float(temp[3]), temp[4])
            data_dict['gps_lat'] = lat
            data_dict['gps_lon'] = lon
            data_dict['abs_north_acc'] = float(temp[7])
            data_dict['abs_east_acc'] = float(temp[6])
            data_dict['abs_up_acc'] = 0
            data_dict['yaw'] = float(temp[9])
            data_list.append(dict(data_dict))

class Kalman():
    def __init__(self, initPos, initVel, gpsSD, accSD, time):
        # Store time.
        self.time = time
        # Matrix to store current state.
        self.currentstate = np.array([[initPos], [initVel]])
        # Error variance matrix for accelerometer.
        self.Q = np.array([[float(accSD * accSD), 0], [0, float(accSD * accSD)]])
        # Error variance matrix for gps.
        self.R = np.array([[float(gpsSD * gpsSD), 0], [0, float(gpsSD * gpsSD)]])
        
        # transformation matrix for input data
        self.H = np.identity(2)
        # initial guess for covariance
        self.P = np.identity(2)
        #identity matrix
        self.I = np.identity(2)

        # accelerometer input matrix
        self.u = np.zeros([1,1])
        # gps input matrix
        self.z = np.zeros([2,1])
        # State transition matrix
        self.A = np.zeros([2,2])
        # Control matrix
        self.B = np.zeros([2,1])

    def predict(self, acc, currtime):
        dTime = self.time  - currtime
        self.updateControlMatrix(dTime)
        self.updateStateMatrix(dTime)
        self.updateAccInputMatrix(acc)
        
        temp1 = np.dot(self.A, self.currentstate)
        temp2 = np.dot(self.B, self.u)
        self.currentstate = np.add(temp1, temp2)


        temp3 = np.dot(self.A, self.P)
        temp4 = np.dot(temp3, self.A.transpose())
        self.P = np.add(temp4, self.Q)


        self.updateTime(currtime)

    def update(self, pos, vel, posError, velError):
        self.z[0,0] = float(pos)
        self.z[1,0] = float(vel)

        if posError != None:
            self.R[0,0] = posError * posError

        self.R[1,1] = velError * velError

        inverse = np.linalg.inv(np.add(self.P, self.R))
        temp = np.dot(self.P, inverse)

        self.currentstate = np.add(self.currentstate, np.dot(temp, np.subtract(self.z, self.currentstate)))
        self.P = np.dot (np.subtract(self.I, temp), self.P)
        

    def updateControlMatrix(self, dTime):
        Time = float(dTime)
        self.B[0,0] = 0.5 * Time * Time
        self.B[1,0] = Time

    def updateStateMatrix(self, dTime):        
        self.A[0,0] = 1.0
        self.A[0,1] = float(dTime)
        self.A[1,0] = 0.0
        self.A[1,1] = 1.0

    def updateAccInputMatrix(self, acc):
        self.u[0,0] = float(acc)

    def updateTime(self, currtime):
        self.time = currtime

    def getPredictedPosition(self):
        return self.currentstate[0,0]

    def getPredictedVelocity(self):
        return self.currentstate[1,0]


readData()

with open('kalmanres.txt', 'w') as f1, open('original.txt', 'w') as f2:

    latlonSD = 2.0 
    altSD = 3.518522417151836

    # got this value by getting standard deviation from accelerometer, assuming that mean SHOULD be 0
    accESD = GRAVITY * 0.033436506994600976
    accNSD = GRAVITY * 0.05355371135598354
    accUSD = GRAVITY * 0.2088683796078286

    initialData = data_list[0]
    lat, lon = LatLonToM(initialData['gps_lat'], initialData['gps_lon'])

    latKalman = Kalman(lat, 0.0, latlonSD, accNSD, initialData['timestamp'])
    lonKalman = Kalman(lon, 0.0, latlonSD, accESD, initialData['timestamp'])
    altKalman = Kalman(initialData['gps_alt'], 0.0, altSD, accUSD, initialData['timestamp'])

    for i in range(1, 20, 1):
        curr_data = data_list[i]
        latKalman.predict(GRAVITY * float(curr_data['abs_north_acc']), curr_data['timestamp'])
        lonKalman.predict(GRAVITY * float(curr_data['abs_east_acc']), curr_data['timestamp'])
        altKalman.predict(GRAVITY * float(curr_data['abs_up_acc']), curr_data['timestamp'])

        if curr_data['gps_lat'] != 0.0:
            lat, lon = LatLonToM(curr_data['gps_lat'], curr_data['gps_lon'])
            lonKalman.update(lon, 0.0, None, 0.0)
            latKalman.update(lat, 0.0, None, 0.0)
            f1.write(str(pLat) + ',' + str(pLon))
            f1.write("\n")
            altKalman.update(curr_data['gps_alt'], 0.0, None, 0.0)
            f2.write(str(curr_data['gps_lat']) + ',' + str(curr_data['gps_lon']))
            f2.write("\n")

        pLatM = latKalman.getPredictedPosition()
        pLonM = lonKalman.getPredictedPosition()
        pAltM = altKalman.getPredictedPosition()

        pLat, pLon = metersToGeopoint(pLatM, pLonM)

        pVelE = lonKalman.getPredictedVelocity()
        pVelN = latKalman.getPredictedVelocity()

        V = math.sqrt(math.pow(pVelE, 2) + math.pow(pVelN, 2))


        
