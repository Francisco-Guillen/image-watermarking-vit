
import bchlib
import numpy as np

class ImprovedECC(object):
    def __init__(self, BCH_POLYNOMIAL=137, BCH_BITS=8, **kwargs):
        """
        ECC melhorado com maior capacidade de correção.
        
        Args:
            BCH_POLYNOMIAL: Polinómio BCH (137 é o padrão)
            BCH_BITS: Número de bits de correção (8 em vez de 5)
        """
        try:
            self.bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
            self.BCH_BITS = BCH_BITS
            print(f"Improved ECC: POLYNOMIAL={BCH_POLYNOMIAL}, BCH_BITS={BCH_BITS}")
        except Exception as e:
            print(f"❌ Erro ao inicializar BCH melhorado: {e}")
            # Fallback para configuração original
            self.bch = bchlib.BCH(137, 5)
            self.BCH_BITS = 5
            print(f"Fallback para ECC original: BCH_BITS=5")
    
    def get_total_len(self):
        return 100  # Manter compatibilidade

    def _encode(self, x):
        # x: 56 bits, {0, 1}, np.array
        # return: 100 bits, {0, 1}, np.array
        dlen = len(x)
        data_str = ''.join(str(x) for x in x.astype(int))
        packet = bytes(int(data_str[i: i + 8], 2) for i in range(0, dlen, 8))
        packet = bytearray(packet)
        ecc = self.bch.encode(packet)
        packet = packet + ecc
        packet = ''.join(format(x, '08b') for x in packet)
        packet = [int(x) for x in packet]
        # Ajustar para 100 bits
        if len(packet) < 100:
            packet.extend([0] * (100 - len(packet)))
        elif len(packet) > 100:
            packet = packet[:100]
        packet = np.array(packet, dtype=np.float32)
        return packet

    def _decode(self, x):
        # x: 100 bits, {0, 1}, np.array
        # return: 56 bits, {0, 1}, np.array
        packet_binary = "".join([str(int(bit)) for bit in x])
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        bitflips = self.bch.decode_inplace(data, ecc)
        if bitflips == -1:  # error, return random data
            data = np.random.binomial(1, .5, 56) 
        else:
            data = ''.join(format(x, '08b') for x in data)
            data = np.array([int(x) for x in data], dtype=np.float32)
        return data  # 56 bits
    
    def _generate(self):
        dlen = 56
        data= np.random.binomial(1, .5, dlen)
        packet = self._encode(data)
        return packet, data

    def generate(self, nsamples=1):
        # generate random 56 bit secret
        data = [self._generate() for _ in range(nsamples)]
        data = (np.array([d[0] for d in data]), np.array([d[1] for d in data]))
        return data  # data with ecc, data org
    
    def _to_text(self, data):
        # data:  {0, 1}, np.array
        # return: str
        data = ''.join([str(int(bit)) for bit in data])
        all_bytes = [ data[i: i+8] for i in range(0, len(data), 8) ]
        text = ''.join([chr(int(byte, 2)) for byte in all_bytes])
        return text.strip()
    
    def _to_binary(self, s):
        if isinstance(s, str):
            out = ''.join([ format(ord(i), "08b") for i in s ])
        elif isinstance(s, bytes):
            out = ''.join([ format(i, "08b") for i in s ])
        elif isinstance(s, np.ndarray) and s.dtype is np.dtype(bool):
            out = ''.join([chr(int(i)) for i in s])
        elif isinstance(s, int) or isinstance(s, np.uint8):
            out = format(s, "08b")
        elif isinstance(s, np.ndarray):
            out = [ format(i, "08b") for i in s ]
        else:
            raise TypeError("Type not supported.")

        return np.array([float(i) for i in out], dtype=np.float32)

    def _encode_text(self, s):
        s = s + ' '*(7-len(s))  # 7 chars
        s = self._to_binary(s)  # 56 bits
        packet = self._encode(s)  # 100 bits
        return packet, s

    def encode_text(self, secret_list, return_pre_ecc=False):
        """encode secret with improved BCH ECC.
        Input: secret (list of strings)
        Output: secret (np array) with shape (B, 100) type float32, val {0,1}"""
        assert np.all(np.array([len(s) for s in secret_list]) <= 7), 'Error! all strings must be less than 7 characters'
        secret_list = [self._encode_text(s) for s in secret_list]
        ecc = np.array([s[0] for s in secret_list], dtype=np.float32)
        if return_pre_ecc:
            return ecc, np.array([s[1] for s in secret_list], dtype=np.float32)
        return ecc
    
    def decode_text(self, data):
        """Decode secret with improved BCH ECC and convert to string.
        Input: secret (torch.tensor) with shape (B, 100) type bool
        Output: secret (B, 56)"""
        data = self.decode(data)
        data = [self._to_text(d) for d in data]
        return data

    def decode(self, data):
        """Decode secret with improved BCH ECC.
        Input: secret (torch.tensor) with shape (B, 100) type bool
        Output: secret (B, 56)"""
        data = data[:, :96]  # Usar apenas os primeiros 96 bits
        data = [self._decode(d) for d in data]
        return np.array(data)
