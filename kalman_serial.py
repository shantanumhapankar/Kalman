import serial
import numpy as np
from datetime import datetime
import math

# Earths gravity
GRAVITY = 9.80665
# Radius in meters
EARTH_RADIUS = 6371 * 1000.0 

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
    return float(deg * math.pi / 180.0)

def RadiansToDegrees(rad):
    return float(rad * 180.0 / math.pi)

def getDistanceM(lat1, lon1, lat2, lon2):
    dlon = degreeToRadians(lon2 - lon1)
    dlat = degreeToRadians(lat2 - lat1)

    a = math.pow(math.sin(dlat/2.0), 2) + math.cos(degreeToRadians(lat1)) * math.cos(degreeToRadians(lat2)) * math.pow(math.sin(dlon/2.0), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    return EARTH_RADIUS * c

# Convert lat and long points from degrees to meters using the haversine formula. 
def LatLonToM(lat, lon):
    latdis = getDistanceM(lat, 0.0, 0.0, 0.0)
    londis = getDistanceM(0.0, lon, 0.0, 0.0)
    if lat < 0: latdis *= -1
    if lon < 0: londis *= -1
    return latdis,londis

# Based on the current lat, long, calculate point ahead at distance 'dist' and angle 'azimuth'
def getPointAhead(lat, lon, dist, azimuth):
    radiusFraction = float(dist / EARTH_RADIUS)

    bearing = degreeToRadians(azimuth)

    lat1 = degreeToRadians(lat)
    lng1 = degreeToRadians(lon)

    lat2_part1 = math.sin(lat1) * math.cos(radiusFraction)
    lat2_part2 = math.cos(lat1) * math.sin(radiusFraction) * math.cos(bearing)
    lat2 = math.asin(lat2_part1 + lat2_part2)

    lng2_part1 = math.sin(bearing) * math.sin(radiusFraction) * math.cos(lat1)
    lng2_part2 = math.cos(radiusFraction) - (math.sin(lat1) * math.sin(lat2))

    lng2 = lng1 + math.atan2(lng2_part1, lng2_part2)
    lng2 = (lng2+3*math.pi) % (2*math.pi) - math.pi

    return (RadiansToDegrees(lat2), RadiansToDegrees(lng2))

def metersToGeopoint(latM, lonM):
    lat, lon = 0.0, 0.0
    # Get point at East
    elat, elon = getPointAhead(lat, lon, lonM, 90.0)
    # Get point at NE
    nelat, nelon = getPointAhead(elat, elon, latM, 0.0)
    return nelat, nelon

def openSer():
    return serial.Serial('/dev/ttyACM0', 9600)

def serData(fd):
    line = fd.readline()
    if  line != None:
        return line.split(',')

def convertSecs(timenow):
    time = timenow.split(':')
    return float(time[0])*3600 + float(time[1])*60 + float(time[2])

class Kalman():
    def __init__(self, initPos, initVel, gpsSD, accSD, time):
        # Store time.
        self.time = time
        # Matrix to store current state.
        self.currentstate = np.array([[initPos], [initVel]], dtype = np.float64)
        # Error variance matrix for accelerometer.
        self.Q = np.array([[float(accSD * accSD), 0], [0, float(accSD * accSD)]], dtype = np.float64)
        # Error variance matrix for gps.
        self.R = np.array([[float(gpsSD * gpsSD), 0], [0, float(gpsSD * gpsSD)]], dtype = np.float64)
        
        # transformation matrix for input data
        self.H = np.identity(2)
        # initial guess for covariance
        self.P = np.identity(2)
        #identity matrix
        self.I = np.identity(2)

        # accelerometer input matrix
        self.u = np.zeros([1,1], dtype = np.float64)
        # gps input matrix
        self.z = np.zeros([2,1], dtype = np.float64)
        # State transition matrix
        self.A = np.zeros([2,2], dtype = np.float64)
        # Control matrix
        self.B = np.zeros([2,1], dtype = np.float64)

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

        y = np.subtract(self.z, self.currentstate)
        s = np.add(self.P, self.R)
        inv = np.linalg.inv(s)
        K = np.dot(self.P, inv)

        temp = np.dot(K, y)
        self.currentstate = np.add(self.currentstate, temp)

        temp2 = np.subtract(self.I, K)
        self.P = np.dot(temp2, self.P)
        

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

def main():
    # Open Serial Port
    serFD = openSer()

    # Define standard deviation for lat, long. 2.0 to be safe.
    latlonSD = 2.0 
    # Accelerometer standard deviation received after few readings from IMU.
    accESD = GRAVITY * 0.033436506994600976
    accNSD = GRAVITY * 0.05355371135598354
    accUSD = GRAVITY * 0.2088683796078286

    # Read serial data but wait for valid lat, long values to initialize the filter.
    datalist = serData(serFD)
    while (float(datalist[0]) == 0.0):
        datalist = serData(serFD)

    timestamp = convertSecs(str(datetime.now().time()))

    # Get lat, long values in degrees and then meters.
    lat, lon = parseDms(float(datalist[0]), datalist[1], float(datalist[2]), datalist[3])
    latM, lonM = LatLonToM(lat, lon)

    # Initialize the kalman filters for lat and long.
    latKalman = Kalman(latM, 0.0, latlonSD, accNSD, timestamp)
    lonKalman = Kalman(lonM, 0.0, latlonSD, accESD, timestamp)

    # Now that kalman is initialized, read data, predict and update in an infinite loop.
    while(1):
        datalist = serData(serFD)
        timestampc = convertSecs(str(datetime.now().time()))
        latKalman.predict(GRAVITY * float(datalist[6]), timestampc)
        lonKalman.predict(GRAVITY * float(datalist[5]), timestampc)

        if float(datalist[0]) != 0.0:
            lat, lon = parseDms(float(datalist[0]), datalist[1], float(datalist[2]), datalist[3])
            latM, lonM = LatLonToM(lat, lon)
            latKalman.update(latM, 0.0, None, 0.0)
            lonKalman.update(lonM, 0.0, None, 0.0)

        pLatM = latKalman.getPredictedPosition()
        pLonM = lonKalman.getPredictedPosition()

        pLat, pLon = metersToGeopoint(pLatM, pLonM)

        

if __name__ == "__main__":
    main()
        
