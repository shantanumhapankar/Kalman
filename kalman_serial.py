import serial
import numpy as np
from datetime import datetime
import math
import pprint

dest_lat = 3518.5730
dest_lat_dir = 'N'
dest_lon = 8044.5298
dest_lon_dir = 'W'
dest_alt = 199.80

# Earths gravity
GRAVITY = 9.80665
# Radius in meters
EARTH_RADIUS = 6371 * 1000.0 

def getBearing(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1;
    y = math.sin(dLon) * math.cos(lat2);
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
    brng = np.rad2deg(math.atan2(y, x));
    if brng < 0: brng+= 360
    return brng

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

    # Define standard deviation for lat, long. 3.0 as per datasheet. 
    latlonSD = 6.0 

    altSD = 10.0

    # Accelerometer standard deviation received after few readings from IMU.
    accNSD = GRAVITY * 0.367107577589
    accESD = GRAVITY * 0.313021102282
    accUSD = GRAVITY * 0.376713286794

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
    altKalman = Kalman(float(datalist[4]), 0.0, altSD, accUSD, timestamp)

    dlat, dlon = parseDms(dest_lat, dest_lat_dir, dest_lon, dest_lon_dir)

    data_dict = {
        "timestamp": 0,
        "gps_lat": 0,
        "gps_lon": 0,
        "gps_alt": 0,
        "pred_lat": 0,
        "pred_lon": 0,
        "pred_alt": 0,
        "abs_north_acc": 0,
        "abs_east_acc": 0,
        "abs_up_acc": 0,
        "mag_heading": 0
    }
    # Now that kalman is initialized, read data, predict and update in an infinite loop.
    with open('kalman.txt', 'w') as f1:
        try:
            while(1):

                datalist = serData(serFD)
                data_dict['timestamp'] = convertSecs(str(datetime.now().time()))
                data_dict['gps_alt'] = float(datalist[4])
                data_dict['abs_east_acc'] = float(datalist[5])
                data_dict['abs_north_acc'] = float(datalist[6]) 
                data_dict['abs_up_acc'] = float(datalist[7])
                data_dict['mag_heading'] = float(datalist[8])
                # print "Kalman Prediction step with accelerometer data."
                # print ''
                latKalman.predict(GRAVITY * data_dict['abs_north_acc'], data_dict['timestamp'])
                lonKalman.predict(GRAVITY *  data_dict['abs_east_acc'], data_dict['timestamp'])
                altKalman.predict(GRAVITY *  data_dict['abs_up_acc'], data_dict['timestamp'])

                if float(datalist[0]) != 0.0:
                    print ''
                    print "Data received from GPS; Kalman Update step."
                    data_dict['gps_lat'], data_dict['gps_lon'] = parseDms(float(datalist[0]), datalist[1], float(datalist[2]), datalist[3])


                    latM, lonM = LatLonToM(data_dict['gps_lat'], data_dict['gps_lon'])
                    latKalman.update(latM, 0.0, None, 0.0)
                    lonKalman.update(lonM, 0.0, None, 0.0)
                    altKalman.update(data_dict['gps_alt'], 0.0, None, 0.0)

                    pp = pprint.PrettyPrinter(indent = 4)
                    pp.pprint(data_dict)
                else:
                    data_dict['gps_lat'] = 0.0
                    data_dict['gps_lon'] = 0.0
                pLatM = latKalman.getPredictedPosition()
                pLonM = lonKalman.getPredictedPosition()
                pAlt = altKalman.getPredictedPosition()
                
                data_dict['pred_lat'], data_dict['pred_lon'] = metersToGeopoint(pLatM, pLonM)
                data_dict['pred_alt'] = pAlt
                print ''
                bearing = getBearing(data_dict['pred_lat'], data_dict['pred_lon'], dlat, dlon)
                print "Distance to destination: " + str(getDistanceM(data_dict['pred_lat'], data_dict['pred_lon'], dlat, dlon)) + " meters."
                print "Angle to steer: Bearing - Heading = " + str(bearing) + " - " + str(data_dict['mag_heading']) + ": " + str(bearing- data_dict['mag_heading'])
                print ''



                f1.write(str(data_dict['gps_lat']) + ',' + str(data_dict['gps_lon']) + ',' + str(data_dict['gps_alt']) + ',' + str(data_dict['pred_lat']) + ',' + str(data_dict['pred_lon']) + ',' + str(data_dict['pred_alt']))
                f1.write('\n')
                # print str(datetime.now().time()) + ' Lat, Long from GPS: (' + str(lat) + ',' + str(data_dict['gps_lon']) + '), ' + 'Kalman predicted Lat, Long: (' + str(pLat) + ',' + str(pLon) + '), ' + 'Acc N, E: (' + str(datalist[5]) + ',' + str(datalist[4]) + ')'
                # print ''
 

        except KeyboardInterrupt:
            print "\n\n"
            print "Ctrl+C; shutting the program."
            f1.flush()

if __name__ == "__main__":
    main()
        
