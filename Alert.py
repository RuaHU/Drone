import numpy as np
from Reader import Reader

def alert(boxes, alert_dict, ps_profiles, threshold:float = 0.7)->bool:
    droneToTrigger = check_alert(boxes, alert_dict, threshold)
    #(ps_profiles)
    """
    A ce stade, je peux faire decoller le drone...
    Je peux lui envoyer les features de la personne à suivre
    Mais pas les poistions de la personnes à suivre à cause du threading.
    Le plus simple serait de passer ces features aux drones qui va les faire remonter dans son thread où il faut ajouter
    la fonction de recherche de personne et ensuite on fait remonter aux drones les instructions de commandes
    mais en faisant ainsi on perdrait en lisibilité.
    Donc à la place, :
    1-  on va supposer que chaque camera drone peut avoir une query personne et modifier cet élément directement depuis alert.
    Et chaque tracker drone se charge de vérifier si il y a une personne à query (sinon se pose et attend).
    OU SINON
    2- On créée une classe personne recherchée pour chaque caméra dans le main (list); on déporte Alerte. On lance un nouveau thread ?
    Question, comment savoir quand arrêter de traquer ? Eviter conditions de blocage
    Mais se pose le proleme de reid car il faut aussi passer la couleur... -> duplication des données <-- Tant pis
    """
    for camId in droneToTrigger:
        print('ALERT HAS BEEN TRIGGERED')
        Reader.knownCameras[camId].triggerAlert() # Risque d appel concurrentiel par 2 threads!!

def check_alert(boxes, alert_dict, threshold:float = 0.7):
    if len(boxes)==0 or len(alert_dict) == 0:
        return set()
    droneToTrigger = set()
    for alert_region in alert_dict:
        covs = covers(boxes, alert_region["zone"])
        # if covs.max() > 0:
        #     print(covs)
        max_covs_index = covs.argmax()
        if covs[max_covs_index] > threshold:
            print("trigger required : prepare to send")
            for droneId in alert_region["droneId"]:
                droneToTrigger.add(droneId)
            #return True, max_covs_index,
    return droneToTrigger

def covers(boxes : np.ndarray,
     box : np.ndarray,
     ious = False)->np.ndarray:
    '''
    inputs:
        box: [x, y, w, h] : np.ndarray
        boxes: shape [N, 4] : np.ndarray
    output:
        the coverage of boxes
    '''
    #print(boxes)
    #print(box)
    b1 = np.array(boxes.copy())
    b2 = np.array(box.copy())
    b1[..., 2:] += b1[..., :2]
    b2[2:] += b2[:2]

    max_xy = np.minimum(b1[:, 2:], b2[2:])
    min_xy = np.maximum(b1[:, :2], b2[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]
    area_b1 = ((b1[:, 2]-b1[:, 0]) * (b1[:, 3]-b1[:, 1]))  # [A,B]
    if ious == False:
        return inter / area_b1

    area_b2 = ((b2[2]-b2[0]) * (b2[3]-b2[1]))  # [A,B]
    union = area_b1 + area_b2 - inter

    if ious:
        return inter / area_b1, inter / union  # [A,B]
