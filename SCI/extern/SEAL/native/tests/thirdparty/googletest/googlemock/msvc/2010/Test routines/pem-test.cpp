// clang++ -DDEBUG=1 -g3 -O0 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -I./ pem-test.cpp -o pem-test.exe ./cryptopp/libcryptopp.a
// clang++ -DNDEBUG=1 -g3 -O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -I./ pem-test.cpp -o pem-test.exe ./cryptopp/libcryptopp.a

#ifdef NDEBUG
# undef NDEBUG
#endif

#include <iostream>
using std::ostream;
using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ios;

#include <string>
using std::string;

#include <cassert>

#include <cryptopp/cryptlib.h>
using CryptoPP::Exception;

#include <cryptopp/dsa.h>
using CryptoPP::DSA;

#include <cryptopp/rsa.h>
using CryptoPP::RSA;

#include <cryptopp/files.h>
using CryptoPP::FileSink;
using CryptoPP::FileSource;

#include <cryptopp/queue.h>
using CryptoPP::ByteQueue;

#include <cryptopp/integer.h>
using CryptoPP::Integer;

#include <cryptopp/eccrypto.h>
using CryptoPP::ECDSA;
using CryptoPP::ECP;
using CryptoPP::EC2N;
using CryptoPP::DL_PublicKey;
using CryptoPP::DL_PrivateKey;
using CryptoPP::DL_PublicKey_EC;
using CryptoPP::DL_PrivateKey_EC;
using CryptoPP::DL_GroupParameters_EC;

#include <cryptopp/gfpcrypt.h>
using CryptoPP::DL_GroupParameters_DSA;

#include <cryptopp/osrng.h>
using CryptoPP::AutoSeededRandomPool;

#include <cryptopp/pem.h>
using CryptoPP::PEM_Load;
using CryptoPP::PEM_Save;

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;
    cin.sync_with_stdio(false);
    
    AutoSeededRandomPool prng;
    
    try
    {
        // Test parsing a PEM file (not recognizing any keys)
        cout << "Running 0" << endl;
        FileSource fs00("rsa-pub.pem", true);
        ByteQueue k0;
        PEM_NextObject(fs00, k0);
        
        FileSink fs01("stuff.pem", true);
        k0.TransferTo(fs01);
        fs01.MessageEnd();
        
        /////
        
        // Test read RSA public
        cout << "Running 1" << endl;
        FileSource fs1("rsa-pub.pem", true);
        RSA::PublicKey k1;
        PEM_Load(fs1, k1);
        
        // Test read RSA private
        cout << "Running 2" << endl;
        FileSource fs2("rsa-priv.pem", true);
        RSA::PrivateKey k2;
        PEM_Load(fs2, k2);
        
        // Test read RSA encrypted private
        cout << "Running 3" << endl;
        FileSource fs3("rsa-enc-priv.pem", true);
        RSA::PrivateKey k3;
        PEM_Load(fs3, k3, "test", 4);
        
        // Test write RSA public
        cout << "Running 4" << endl;
        FileSink fs4("rsa-pub-xxx.pem", true);
        PEM_Save(fs4, k1);
        fs4.MessageEnd();
        
        // Test write RSA private
        cout << "Running 5" << endl;
        FileSink fs5("rsa-priv-xxx.pem", true);
        PEM_Save(fs5, k2);
        fs5.MessageEnd();
        
        // Test write RSA encrypted private
        cout << "Running 6" << endl;
        FileSink fs6("rsa-enc-priv-xxx.pem", true);
        PEM_Save(fs6, k3, "AES-128-CBC", "test", 4);
        fs6.MessageEnd();
        
        /////
        
        // Test read DSA public
        cout << "Running 7" << endl;
        FileSource fs7("dsa-pub.pem", true);
        DSA::PublicKey k7;
        PEM_Load(fs7, k7);
        
        // Test read DSA private
        cout << "Running 8" << endl;
        FileSource fs8("dsa-priv.pem", true);
        DSA::PrivateKey k8;
        PEM_Load(fs8, k8);
        
        // Test read DSA encrypted private
        cout << "Running 9" << endl;
        FileSource fs9("dsa-enc-priv.pem", true);
        DSA::PrivateKey k9;
        PEM_Load(fs9, k9, "test", 4);
        
        // Test write DSA public
        cout << "Running 10" << endl;
        FileSink fs10("dsa-pub-xxx.pem", true);
        PEM_Save(fs10, k7);
        fs10.MessageEnd();
        
        // Test write DSA private
        cout << "Running 11" << endl;
        FileSink fs11("dsa-priv-xxx.pem", true);
        PEM_Save(fs11, k8);
        fs11.MessageEnd();
        
        // Test write DSA encrypted private
        cout << "Running 12" << endl;
        FileSink fs12("dsa-enc-priv-xxx.pem", true);
        PEM_Save(fs12, k9, "AES-128-CBC", "test", 4);
        fs12.MessageEnd();
        
        /////
        
        // Test read EC public
        cout << "Running 13" << endl;
        FileSource fs13("ec-pub.pem", true);
        DL_PublicKey_EC<ECP> k13;
        PEM_Load(fs13, k13);
        
        // Test read EC private
        cout << "Running 14" << endl;
        FileSource fs14("ec-priv.pem", true);
        DL_PrivateKey_EC<ECP> k14;
        PEM_Load(fs14, k14);
        
        // Test read EC encrypted private
        cout << "Running 15" << endl;
        FileSource fs15("ec-enc-priv.pem", true);
        DL_PrivateKey_EC<ECP> k15;
        PEM_Load(fs15, k15, "test", 4);
        
        // Test write EC public
        cout << "Running 16" << endl;
        FileSink fs16("ec-pub-xxx.pem", true);
        PEM_Save(fs16, k13);
        fs16.MessageEnd();
        
        // Test write EC private
        cout << "Running 17" << endl;
        FileSink fs17("ec-priv-xxx.pem", true);
        PEM_Save(fs17, k14);
        fs17.MessageEnd();
        
        // Test read EC encrypted private
        cout << "Running 18" << endl;
        FileSink fs18("ec-enc-priv-xxx.pem", true);
        PEM_Save(fs18, k15, "AES-128-CBC", "test", 4);
        fs18.MessageEnd();
        
        /////
        
        // Two public keys in this file
        cout << "Running 19" << endl;
        FileSource fs19("rsa-pub-double.pem", true);
        RSA::PublicKey k19a, k19b;
        PEM_Load(fs19, k19a);
        PEM_Load(fs19, k19b);
        assert(k19a.GetModulus() == k19b.GetModulus() && k19a.GetPublicExponent() == k19b.GetPublicExponent());
        
        // Diffie-Hellman
        
        cout << "Running 20" << endl;
        FileSource fs20("dh-params.pem", true);
        Integer i1, i2;
        PEM_DH_Load(fs20, i1, i2);
        
        cout << "Running 21" << endl;
        FileSink fs21("dh-params-xxx.pem", true);
        PEM_DH_Save(fs21, i1, i2);
        fs21.MessageEnd();
        
        cout << "Running 22" << endl;
        FileSource fs22("dh-params-xxx.pem", true);
        PEM_DH_Load(fs22, i1, i2);
        
        // Missing CR or LF on last line
        
        cout << "Running 23" << endl;
        FileSource fs23("rsa-trunc-1.pem", true);
        RSA::PublicKey k23;
        PEM_Load(fs23, k23);
        
        try
        {
            // Missing two characters - the last dash and the LF
            
            cout << "Running 24" << endl;
            FileSource fs24("rsa-trunc-2.pem", true);
            RSA::PublicKey k24;
            PEM_Load(fs24, k24);
            
            cerr << "Failed test 24" << endl;
            
        }
        catch(const Exception& ex)
        {
            // Do nothing... expected
        }
        
        try
        {
            // Only the "-----BEGINE PUBLIC KEY-----"
            
            cout << "Running 25" << endl;
            FileSource fs25("rsa-short.pem", true);
            RSA::PublicKey k25;
            PEM_Load(fs25, k25);
            
            cerr << "Failed test 25" << endl;
            
        }
        catch(const Exception& ex)
        {
            // Do nothing... expected
        }
        
        // Two keys concat'd, missing the CRLF between them
        
        // Two public keys in this file
        cout << "Running 26" << endl;
        FileSource fs26("rsa-concat.pem", true);
        RSA::PublicKey k26a, k26b;
        PEM_Load(fs26, k26a);
        PEM_Load(fs26, k26b);
        assert(k26a.GetModulus() == k26b.GetModulus() && k26a.GetPublicExponent() == k26b.GetPublicExponent());
        
        // DSA params
        cout << "Running 27" << endl;
        FileSource fs27("dsa-params.pem", true);
        DL_GroupParameters_DSA p27;
        PEM_Load(fs27, p27);
        
        cout << "Running 28" << endl;
        FileSink fs28("dsa-params-xxx.pem", true);
        PEM_Save(fs28, p27);
        fs28.MessageEnd();
        
        // EC params
        cout << "Running 29" << endl;
        FileSource fs29("ec-params.pem", true);
        DL_GroupParameters_EC<ECP> p29;
        PEM_Load(fs29, p29);
        
        cout << "Running 30" << endl;
        FileSink fs30("ec-params-xxx.pem", true);
        PEM_Save(fs30, p29);
        fs30.MessageEnd();
        
        // cacert.pem has about 150 certs in it
        cout << "Running 31" << endl;
        FileSource fs31("cacert.pem", true);
        ByteQueue temp;
        unsigned count = 0;
        
        try
        {
            while(fs31.AnyRetrievable())
            {
                PEM_NextObject(fs31, temp);
                temp.Clear();
                count++;
            }
        }
        catch(const Exception& ex)
        {
            cerr << "Failed test 31" << endl;
        }
        
        cout << "Parsed " << count << " certifcates from cacert.pem" << endl;
        
        cout << "All tests passed" << endl;
        
    }
    catch(const Exception& ex)
    {
        cerr << ex.what() << endl;
    }
    
    return 0;
}
